"""Run pipelines over a question split and aggregate metrics with bootstrap confidence intervals."""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from analyst.corpus.retriever import BM25Retriever
from analyst.eval.judge import QuestionScore, load_facts, score_answer
from analyst.guardrails import GuardrailViolation
from analyst.llm.base import LLMClient
from analyst.pipelines import MultiAgentPipeline, SingleAgentPipeline
from analyst.pipelines.multi_agent import MultiAgentConfig
from analyst.types import Answer

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


def evaluate(
    system: str,
    llm: LLMClient,
    retriever: BM25Retriever,
    questions: list[dict],
    facts_path: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    facts = load_facts(facts_path)
    pipe = SYSTEMS[system](llm, retriever)
    scores: list[QuestionScore] = []
    calls, tokens, costs, lats = [], [], [], []
    errors = 0
    for i, q in enumerate(questions):
        try:
            res = pipe.run(q["question"], trace_id=f"{system}-{q['id']}")
            answer, st = res.answer, res.stats
            calls.append(st.llm_calls)
            tokens.append(st.input_tokens + st.output_tokens)
            costs.append(st.cost_usd)
            lats.append(st.latency_s)
        except (GuardrailViolation, RuntimeError):
            errors += 1
            answer = Answer(question=q["question"], summary="", claims=[])
        scores.append(score_answer(q["id"], answer, q["required"], facts))
        if progress and (i + 1) % 25 == 0:
            progress(f"{system}: {i + 1}/{len(questions)}")
    out: dict[str, Any] = {
        "system": system,
        "n_questions": len(questions),
        "errors": errors,
        "n_claims": sum(s.n_claims for s in scores),
    }
    for name, fn in METRICS.items():
        lo, hi = _bootstrap(scores, fn)
        out[name] = {"value": round(fn(scores), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["cost"] = {
        "mean_llm_calls": round(statistics.mean(calls), 2) if calls else 0,
        "mean_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
        "mean_cost_usd": round(statistics.mean(costs), 5) if costs else 0,
        "latency_p50_s": round(_pct(lats, 0.5), 3),
        "latency_p95_s": round(_pct(lats, 0.95), 3),
    }
    out["hallucination_examples"] = [h for s in scores for h in s.hallucinations][:5]
    out["per_question"] = [asdict(s) for s in scores]
    return out
