"""Command-line entry point: `analyst ask | eval | trace | serve | build-data`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from analyst.config import data_dir
from analyst.corpus.retriever import BM25Retriever
from analyst.llm import make_client
from analyst.tracing import render_tree

OFFLINE_NOTE = (
    "> **Offline run (simulated LLM).** These numbers come from `MockLLM`, a seeded behavioural "
    "simulator with *chosen* error rates (see `src/analyst/llm/mock.py`). They verify that the "
    "pipeline, critic loop, guardrails and judge behave as designed. They are **not** a "
    "measurement of any real model. Run `make eval-live` with an API key for real numbers."
)
LIVE_NOTE = "> **Live run** against `{model}` (effort `{effort}`)."


def _pipeline(name: str, llm, retriever):
    from analyst.eval.harness import SYSTEMS

    return SYSTEMS[name](llm, retriever)


def cmd_ask(a: argparse.Namespace) -> int:
    retriever = BM25Retriever.from_jsonl(data_dir() / "corpus.jsonl")
    llm = make_client(a.provider)
    pipe = _pipeline(a.pipeline, llm, retriever)
    res = pipe.run(a.question)
    print(res.answer.summary)
    print(
        f"\n{res.stats.llm_calls} LLM calls · {res.stats.input_tokens}+"
        f"{res.stats.output_tokens} tokens · est. ${res.stats.cost_usd:.4f} · "
        f"{res.stats.latency_s:.2f}s"
    )
    if res.answer.guardrail_events:
        print("guardrails:", "; ".join(res.answer.guardrail_events))
    if a.trace_dir:
        path = pipe.last_tracer.save(a.trace_dir)
        print(f"trace: {path}")
    if a.json:
        print(res.model_dump_json(indent=2))
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from analyst.eval.harness import SYSTEMS, evaluate, load_questions
    from analyst.eval.report import merge_seeds, render_markdown

    d = data_dir()
    retriever = BM25Retriever.from_jsonl(d / "corpus.jsonl")
    questions = load_questions(d / f"questions_{a.split}.jsonl", a.limit or None)
    systems = a.systems.split(",") if a.systems else list(SYSTEMS)
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown system(s): {unknown}; choose from {list(SYSTEMS)}")
    provider = a.provider or "mock"
    seeds = list(range(a.seeds)) if provider == "mock" else [0]
    # The seeded mock must stay sequential to be reproducible; live runs go in parallel.
    workers = 1 if provider == "mock" else a.workers
    run_info: dict[str, object] = {
        "provider": provider,
        "split": a.split,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "workers": workers,
    }
    if provider != "mock":
        live = make_client(provider)
        run_info["model"] = live.model
        run_info["effort"] = getattr(live, "effort", "n/a")
    results = []
    for system in systems:
        runs = []
        for seed in seeds:
            llm = make_client(provider, seed=seed) if provider == "mock" else make_client(provider)
            runs.append(
                evaluate(
                    system,
                    llm,
                    retriever,
                    questions,
                    d / "facts.jsonl",
                    progress=lambda m: print(m, file=sys.stderr),
                    workers=workers,
                    trace_dir=Path(a.trace_dir) / system if a.trace_dir else None,
                )
            )
        merged = merge_seeds(runs)
        if provider != "mock":
            merged["run"] = run_info
        results.append(merged)
        print(
            f"{system:28s} halluc={merged['hallucinated_claim_rate']['value']:.3f} "
            f"recall={merged['fact_recall']['value']:.3f} "
            f"abstain={merged['abstention_accuracy']['value']:.3f} "
            f"acc={merged['question_accuracy']['value']:.3f} "
            f"errors={merged['errors']} {merged['error_types']} "
            f"json_repairs={merged['json_repairs']} fallbacks={merged['fallbacks']}",
            file=sys.stderr,
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{provider}_{a.split}" + (f"_{a.tag}" if a.tag else "")
    (out / f"{stem}.json").write_text(json.dumps(results, indent=2))
    if provider == "mock":
        note = (
            OFFLINE_NOTE
            + f"\n\nSplit `{a.split}`, {len(questions)} questions × {len(seeds)} seeds."
        )
    else:
        spend = sum(r["cost"].get("total_cost_usd", 0.0) for r in results)
        note = (
            LIVE_NOTE.format(model=run_info["model"], effort=run_info["effort"])
            + f" Split `{a.split}`, {len(questions)} questions, run {run_info['date']}, "
            f"{workers} questions in parallel, total spend ${spend:.2f} (from token counts)."
        )
    # A relative-improvement headline is only meaningful for a real model.
    (out / f"{stem}.md").write_text(
        render_markdown(results, f"Evaluation: {stem}", note, show_relative=provider != "mock")
    )
    print(f"wrote {out / stem}.md and .json", file=sys.stderr)
    return 0


def cmd_readme_table(a: argparse.Namespace) -> int:
    from analyst.eval.report import render_readme_live

    print(render_readme_live(json.loads(Path(a.path).read_text())), end="")
    return 0


def cmd_trace(a: argparse.Namespace) -> int:
    print(render_tree(json.loads(Path(a.path).read_text())))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("analyst.api:app", host=a.host, port=a.port)
    return 0


def cmd_build_data(a: argparse.Namespace) -> int:
    from analyst.corpus.generate import build

    build(Path(a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyst", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--pipeline", default="multi_agent")
    s.add_argument("--provider", default=None, help="mock (default) or anthropic")
    s.add_argument("--trace-dir", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="run the evaluation harness")
    s.add_argument("--split", default="test", choices=["dev", "test", "adversarial"])
    s.add_argument("--systems", default=None, help="comma-separated; default all")
    s.add_argument("--provider", default=None)
    s.add_argument("--seeds", type=int, default=5, help="mock only")
    s.add_argument("--limit", type=int, default=None, help="first N questions; 0 or unset = all")
    s.add_argument("--out", default="results")
    s.add_argument("--tag", default="", help="suffix for output files, e.g. 'ablation'")
    s.add_argument("--workers", type=int, default=4, help="parallel questions (live only)")
    s.add_argument("--trace-dir", default=None, help="save one trace JSON per question")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("readme-table", help="render the README table for a live results JSON")
    s.add_argument("path")
    s.set_defaults(fn=cmd_readme_table)

    s = sub.add_parser("trace", help="pretty-print a saved trace")
    s.add_argument("path")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-data", help="regenerate the synthetic benchmark")
    s.add_argument("--out", default="data")
    s.set_defaults(fn=cmd_build_data)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
