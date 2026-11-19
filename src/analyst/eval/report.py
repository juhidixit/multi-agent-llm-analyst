"""Aggregate runs across seeds (question-clustered bootstrap) and render a Markdown report."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from analyst.eval.harness import METRICS, _bootstrap
from analyst.eval.judge import QuestionScore

COUNT_FIELDS = [
    "n_claims",
    "n_hallucinated",
    "answerable",
    "answerable_correct",
    "unanswerable",
    "unanswerable_abstained",
]


def merge_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-question counts across seeds, then bootstrap over *questions* (clusters)."""
    by_q: dict[str, QuestionScore] = {}
    for run in runs:
        for pq in run["per_question"]:
            s = by_q.setdefault(pq["qid"], QuestionScore(qid=pq["qid"]))
            for f in COUNT_FIELDS:
                setattr(s, f, getattr(s, f) + pq[f])
    # question_accuracy is averaged per run (it is not additive across seeds).
    pooled = list(by_q.values())
    out: dict[str, Any] = {
        "system": runs[0]["system"],
        "seeds": len(runs),
        "n_questions": runs[0]["n_questions"],
        "n_claims": sum(r["n_claims"] for r in runs),
        "errors": sum(r["errors"] for r in runs),
        "error_types": dict(sum((Counter(r.get("error_types", {})) for r in runs), Counter[str]())),
        "json_repairs": sum(r.get("json_repairs", 0) for r in runs),
        "fallbacks": sum(r.get("fallbacks", 0) for r in runs),
        "n_unanswerable_facts": sum(s.unanswerable for s in pooled),
    }
    for name, fn in METRICS.items():
        if name == "question_accuracy" and len(runs) == 1:
            # One run (live model): a question-level bootstrap interval, as for the others.
            out[name] = dict(runs[0][name])
            continue
        if name == "question_accuracy":
            vals = [r[name]["value"] for r in runs]
            out[name] = {
                "value": round(statistics.mean(vals), 4),
                "seed_min": min(vals),
                "seed_max": max(vals),
            }
            continue
        lo, hi = _bootstrap(pooled, fn)
        out[name] = {
            "value": round(fn(pooled), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "per_seed": [r[name]["value"] for r in runs],
        }
    out["cost"] = {
        k: round(statistics.mean(r["cost"][k] for r in runs), 5) for k in runs[0]["cost"]
    }
    if "total_cost_usd" in out["cost"]:
        out["cost"]["total_cost_usd"] = round(sum(r["cost"]["total_cost_usd"] for r in runs), 4)
    out["hallucination_examples"] = runs[0]["hallucination_examples"]
    return out


def _fmt(m: dict[str, Any]) -> str:
    if "ci95" in m:
        return f"{m['value'] * 100:.1f}% ({m['ci95'][0] * 100:.1f}–{m['ci95'][1] * 100:.1f})"
    return f"{m['value'] * 100:.1f}% ({m['seed_min'] * 100:.1f}–{m['seed_max'] * 100:.1f})"


def render_markdown(
    results: list[dict[str, Any]], title: str, note: str, show_relative: bool = True
) -> str:
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | "
        "Question accuracy ↑ | LLM calls / q | Tokens / q | Est. cost / q | p50 / p95 latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["cost"]
        lines.append(
            f"| `{r['system']}` | {_fmt(r['hallucinated_claim_rate'])} | {_fmt(r['fact_recall'])}"
            f" | {_fmt(r['abstention_accuracy']) if r['n_unanswerable_facts'] else 'n/a'} | {_fmt(r['question_accuracy'])} | "
            f"{c['mean_llm_calls']:.2f} | {c['mean_tokens']:.0f} | ${c['mean_cost_usd']:.4f} | "
            f"{c['latency_p50_s']:.2f}s / {c['latency_p95_s']:.2f}s |"
        )
    base = next((r for r in results if r["system"] == "single_agent"), None)
    multi = next((r for r in results if r["system"] == "multi_agent"), None)
    if show_relative and base and multi and base["hallucinated_claim_rate"]["value"] > 0:
        b = base["hallucinated_claim_rate"]["value"]
        m = multi["hallucinated_claim_rate"]["value"]
        lines += [
            "",
            f"Relative change in hallucinated-claim rate, `multi_agent` vs "
            f"`single_agent`: **{(m - b) / b * 100:+.1f}%** "
            f"({b * 100:.1f}% → {m * 100:.1f}%).",
        ]
    if any("error_types" in r for r in results):
        lines += ["", "Reliability (all seeds pooled):", ""]
        for r in results:
            lines.append(f"- `{r['system']}`: {_reliability(r)}")
    multi_seed = any(r.get("seeds", 1) > 1 for r in results)
    lines += [
        "",
        "Brackets are 95% bootstrap intervals, resampling questions (claims from the "
        "same question are correlated, so questions are the unit)."
        + (" For question accuracy the bracket is the min–max across seeds." if multi_seed else ""),
    ]
    return "\n".join(lines) + "\n"


def _reliability(r: dict[str, Any]) -> str:
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(r.get("error_types", {}).items()))
    return (
        f"{r['errors']} failed questions"
        + (f" ({kinds})" if kinds else "")
        + f", {r.get('json_repairs', 0)} JSON repair retries, "
        f"{r.get('fallbacks', 0)} fallback-model calls"
    )


README_HEADER = (
    "| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | "
    "Question accuracy ↑ | LLM calls / q | Cost / q | p50 / p95 latency |\n"
    "|---|---|---|---|---|---|---|---|"
)


def render_readme_live(results: list[dict[str, Any]]) -> str:
    """The exact README block for one live results file.

    ``tests/test_readme_results.py`` checks that the README contains this block verbatim, so
    every live number in the README is traceable to a committed results JSON.
    """
    rows = [README_HEADER]
    for r in results:
        c = r["cost"]
        abst = _fmt(r["abstention_accuracy"]) if r["n_unanswerable_facts"] else "n/a"
        rows.append(
            f"| `{r['system']}` | {_fmt(r['hallucinated_claim_rate'])} | "
            f"{_fmt(r['fact_recall'])} | {abst} | {_fmt(r['question_accuracy'])} | "
            f"{c['mean_llm_calls']:.2f} | ${c['mean_cost_usd']:.4f} | "
            f"{c['latency_p50_s']:.1f}s / {c['latency_p95_s']:.1f}s |"
        )
    run = results[0].get("run", {})
    spend = sum(r["cost"].get("total_cost_usd", 0.0) for r in results)
    rows += [
        "",
        f"Model `{run.get('model', '?')}`, effort `{run.get('effort', '?')}`, split "
        f"`{run.get('split', '?')}` ({results[0]['n_questions']} questions), run "
        f"{run.get('date', '?')}, spend ${spend:.2f}.",
    ]
    rows += [f"Reliability, `{r['system']}`: {_reliability(r)}." for r in results]
    return "\n".join(rows) + "\n"
