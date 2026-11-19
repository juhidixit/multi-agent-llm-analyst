"""Run pipelines over a question split and aggregate metrics with bootstrap confidence intervals."""

from __future__ import annotations

import json
import random
import statistics
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from analyst.corpus.retriever import BM25Retriever
from analyst.eval.judge import QuestionScore, load_facts, score_answer
from analyst.guardrails import GuardrailViolation
from analyst.llm.base import LLMClient
from analyst.pipelines import MultiAgentPipeline, SingleAgentPipeline
from analyst.pipelines.multi_agent import MultiAgentConfig
from analyst.types import Answer, RunStats

SYSTEMS: dict[str, Callable[[LLMClient, BM25Retriever], Any]] = {
    "single_agent": lambda llm, r: SingleAgentPipeline(llm, r),
    "multi_agent": lambda llm, r: MultiAgentPipeline(llm, r),
    "multi_agent_no_critic": lambda llm, r: MultiAgentPipeline(
        llm, r, MultiAgentConfig(max_critic_rounds=0)
    ),
    "multi_agent_no_guardrails": lambda llm, r: MultiAgentPipeline(
        llm, r, MultiAgentConfig(use_guardrails=False)
    ),
}


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _bootstrap(
    scores: list[QuestionScore],
    fn: Callable[[list[QuestionScore]], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(scores) for _ in scores]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def _halluc_rate(ss: list[QuestionScore]) -> float:
    n = sum(s.n_claims for s in ss)
    return sum(s.n_hallucinated for s in ss) / n if n else 0.0


def _recall(ss: list[QuestionScore]) -> float:
    n = sum(s.answerable for s in ss)
    return sum(s.answerable_correct for s in ss) / n if n else 0.0


def _abstain(ss: list[QuestionScore]) -> float:
    n = sum(s.unanswerable for s in ss)
    return sum(s.unanswerable_abstained for s in ss) / n if n else 0.0


def _accuracy(ss: list[QuestionScore]) -> float:
    return sum(s.fully_correct for s in ss) / len(ss) if ss else 0.0


METRICS = {
    "hallucinated_claim_rate": _halluc_rate,
    "fact_recall": _recall,
    "abstention_accuracy": _abstain,
    "question_accuracy": _accuracy,
}


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


class TooManyErrors(RuntimeError):
    """Raised to stop a run early when every recent question failed (bad key, outage)."""


def classify_error(exc: BaseException) -> str:
    """Bucket a per-question failure so a live run can be audited (refusal, budget, JSON...)."""
    from analyst.agents.base import AgentOutputError
    from analyst.guardrails import BudgetExceeded

    if isinstance(exc, GuardrailViolation):
        return "guardrail"
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, AgentOutputError):
        return "json_parse"
    if type(exc).__name__ == "RefusalError":
        return "refusal"
    if type(exc).__module__.startswith("anthropic"):
        return "api_error"
    return "other"


def evaluate(
    system: str,
    llm: LLMClient,
    retriever: BM25Retriever,
    questions: list[dict],
    facts_path: Path,
    progress: Callable[[str], None] | None = None,
    workers: int = 1,
    trace_dir: Path | None = None,
    max_consecutive_errors: int = 10,
) -> dict[str, Any]:
    """Score one system on ``questions``.

    ``workers > 1`` runs questions concurrently (one pipeline per thread), which a live
    provider needs to finish a full split in reasonable time. Keep ``workers=1`` for the
    seeded mock so results stay reproducible.
    """
    facts = load_facts(facts_path)
    local = threading.local()
    lock = threading.Lock()
    state = {"done": 0, "consecutive_errors": 0}

    def run_one(q: dict) -> tuple[Answer, RunStats | None, str | None, dict[str, float]]:
        if not hasattr(local, "pipe"):
            local.pipe = SYSTEMS[system](llm, retriever)
        pipe = local.pipe
        err: str | None = None
        st: RunStats | None = None
        counters: dict[str, float] = {"json_repairs": 0, "fallbacks": 0, "cost_usd": 0.0}
        pipe.last_tracer = None
        try:
            res = pipe.run(q["question"], trace_id=f"{system}-{q['id']}")
            answer, st = res.answer, res.stats
        except Exception as exc:  # noqa: BLE001 - every failure is scored and counted
            err = classify_error(exc)
            answer = Answer(question=q["question"], summary="", claims=[])
        tracer = getattr(pipe, "last_tracer", None)
        if tracer is not None:
            counters = {
                "json_repairs": tracer.json_repairs,
                "fallbacks": tracer.fallbacks,
                "cost_usd": tracer.cost_usd,  # includes spend on questions that failed
            }
            if trace_dir is not None and tracer.spans:
                tracer.save(trace_dir)
        with lock:
            state["done"] += 1
            state["consecutive_errors"] = state["consecutive_errors"] + 1 if err else 0
            if progress and state["done"] % 25 == 0:
                progress(f"{system}: {state['done']}/{len(questions)}")
            if state["consecutive_errors"] >= max_consecutive_errors and err in (
                "api_error",
                "other",
            ):
                raise TooManyErrors(
                    f"{system}: {max_consecutive_errors} questions in a row failed ({err}); "
                    "stopping. Check the API key, model name and account limits."
                )
        return answer, st, err, counters

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(run_one, questions))
    else:
        outcomes = [run_one(q) for q in questions]

    scores: list[QuestionScore] = []
    calls, tokens, costs, lats = [], [], [], []
    error_types: Counter[str] = Counter()
    json_repairs = fallbacks = 0
    total_cost = 0.0
    for q, (answer, st, err, counters) in zip(questions, outcomes, strict=True):
        if err:
            error_types[err] += 1
        if st is not None:
            calls.append(st.llm_calls)
            tokens.append(st.input_tokens + st.output_tokens)
            costs.append(st.cost_usd)
            lats.append(st.latency_s)
        json_repairs += int(counters["json_repairs"])
        fallbacks += int(counters["fallbacks"])
        total_cost += counters["cost_usd"]
        scores.append(score_answer(q["id"], answer, q["required"], facts))
    out: dict[str, Any] = {
        "system": system,
        "n_questions": len(questions),
        "errors": sum(error_types.values()),
        "error_types": dict(error_types),
        "json_repairs": json_repairs,
        "fallbacks": fallbacks,
        "n_claims": sum(s.n_claims for s in scores),
    }
    for name, fn in METRICS.items():
        lo, hi = _bootstrap(scores, fn)
        out[name] = {"value": round(fn(scores), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["cost"] = {
        "mean_llm_calls": round(statistics.mean(calls), 2) if calls else 0,
        "mean_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
        "mean_cost_usd": round(statistics.mean(costs), 5) if costs else 0,
        "total_cost_usd": round(total_cost, 4),
        "latency_p50_s": round(_pct(lats, 0.5), 3),
        "latency_p95_s": round(_pct(lats, 0.95), 3),
    }
    out["hallucination_examples"] = [h for s in scores for h in s.hallucinations][:5]
    out["per_question"] = [asdict(s) for s in scores]
    return out
