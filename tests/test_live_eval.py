"""Live-run plumbing, tested without an API key: parallel runs, error buckets, the client."""

from types import SimpleNamespace

import pytest

from analyst.config import data_dir
from analyst.eval.harness import TooManyErrors, evaluate, load_questions
from analyst.eval.report import merge_seeds, render_readme_live
from analyst.llm.base import LLMRequest, LLMResponse
from analyst.llm.pricing import cost_usd


class Raising:
    model = "claude-opus-5-5"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise self.exc


class NotJson:
    model = "claude-opus-5-5"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="Sure! Here is the answer.", input_tokens=100, output_tokens=10, model=self.model
        )


def _qs(n: int = 12) -> list[dict]:
    return load_questions(data_dir() / "questions_dev.jsonl", limit=n)


def test_parallel_run_matches_sequential(perfect_llm, retriever):
    d = data_dir()
    seq = evaluate("multi_agent", perfect_llm, retriever, _qs(20), d / "facts.jsonl", workers=1)
    par = evaluate("multi_agent", perfect_llm, retriever, _qs(20), d / "facts.jsonl", workers=4)
    for key in ("hallucinated_claim_rate", "fact_recall", "question_accuracy"):
        assert par[key]["value"] == seq[key]["value"]
    assert par["errors"] == 0 and par["json_repairs"] == 0


def test_refusals_are_counted_not_crashed(retriever):
    from analyst.llm.anthropic_client import RefusalError

    r = evaluate(
        "single_agent",
        Raising(RefusalError("no")),
        retriever,
        _qs(),
        data_dir() / "facts.jsonl",
        max_consecutive_errors=1000,
    )
    assert r["error_types"] == {"refusal": 12}


def test_json_failures_are_counted(retriever):
    r = evaluate("single_agent", NotJson(), retriever, _qs(5), data_dir() / "facts.jsonl")
    assert r["error_types"] == {"json_parse": 5}
    assert r["json_repairs"] == 10  # two attempts per question
    # Spend on failed questions still counts toward the total.
    assert r["cost"]["total_cost_usd"] == pytest.approx(
        5 * 2 * cost_usd("claude-opus-5-5", 100, 10), abs=1e-4
    )


def test_run_stops_when_every_question_fails(retriever):
    with pytest.raises(TooManyErrors):
        evaluate(
            "single_agent",
            Raising(ValueError("boom")),
            retriever,
            _qs(),
            data_dir() / "facts.jsonl",
            max_consecutive_errors=5,
        )


def test_saves_one_trace_per_question(perfect_llm, retriever, tmp_path):
    evaluate(
        "single_agent",
        perfect_llm,
        retriever,
        _qs(4),
        data_dir() / "facts.jsonl",
        trace_dir=tmp_path,
    )
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_readme_block_uses_bootstrap_for_single_live_run(perfect_llm, retriever):
    r = merge_seeds(
        [evaluate("single_agent", perfect_llm, retriever, _qs(6), data_dir() / "facts.jsonl")]
    )
    r["run"] = {"model": "claude-opus-5-5", "effort": "low", "split": "dev", "date": "2026-01-01"}
    assert "ci95" in r["question_accuracy"]
    block = render_readme_live([r])
    assert "Model `claude-opus-5-5`, effort `low`" in block and "| `single_agent` |" in block


def test_pricing_handles_snapshots_and_fallback_models():
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == 4.0
    assert cost_usd("claude-opus-4-8", 0, 1_000_000) == 25.0
    assert cost_usd("claude-opus-5-20260101", 1_000_000, 0) == 5.0


def test_anthropic_client_request_shape(monkeypatch):
    pytest.importorskip("anthropic")
    from analyst.llm.anthropic_client import AnthropicLLM, RefusalError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("ANALYST_MODEL", raising=False)
    llm = AnthropicLLM()
    sent: dict = {}

    def fake_create(**kw):
        sent.update(kw)
        return SimpleNamespace(
            stop_reason=kw.pop("_stop", "end_turn"),
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text='{"ok": true}'),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-opus-5-5",
        )

    llm._client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    )
    resp = llm.complete(LLMRequest(role="planner", system="s", prompt="p"))
    assert resp.text == '{"ok": true}' and resp.model == "claude-opus-5-5" and not resp.fallback
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["fallbacks"] == "default"
    assert "thinking" not in sent  # Opus 5.5 rejects disabled thinking; effort is the control

    def refuse(**kw):
        return SimpleNamespace(stop_reason="refusal", content=[], usage=None, model="x")

    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=refuse)))
    with pytest.raises(RefusalError):
        llm.complete(LLMRequest(role="critic", system="s", prompt="p"))
