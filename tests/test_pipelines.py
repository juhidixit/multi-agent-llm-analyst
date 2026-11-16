import pytest

from analyst.agents.base import AgentOutputError, extract_json
from analyst.guardrails import Budget
from analyst.llm.base import LLMResponse
from analyst.pipelines import MultiAgentPipeline, SingleAgentPipeline
from analyst.pipelines.multi_agent import MultiAgentConfig

Q_COMPARE = (
    "Which company had higher fiscal 2024 revenue, Jessamine Analytics or "
    "Ixora Dynamics, and by how much?"
)


def test_multi_agent_answers_comparison_with_computed_difference(perfect_llm, retriever):
    res = MultiAgentPipeline(perfect_llm, retriever).run(Q_COMPARE)
    assert len(res.answer.claims) == 2
    assert "(computed)" in res.answer.summary
    assert all(c.citations for c in res.answer.claims)
    assert res.stats.llm_calls == 4  # planner + 2 researchers + 1 critic round


def test_unknown_company_abstains(perfect_llm, retriever):
    res = MultiAgentPipeline(perfect_llm, retriever).run("Who is the current CEO of Quorvex Group?")
    assert not res.answer.claims and res.answer.abstentions


def test_critic_rejection_is_never_shipped(retriever):
    """With a critic that rejects everything, the pipeline must abstain rather than answer."""
    from analyst.llm.mock import MockLLM

    class RejectAll(MockLLM):
        def _role_critic(self, payload):
            return {
                "verdicts": [
                    {"claim_id": c["id"], "supported": False, "feedback": "no"}
                    for c in payload["claims"]
                ]
            }

    res = MultiAgentPipeline(RejectAll(p_misread=0), retriever).run(
        "Where is Lumora Systems headquartered?"
    )
    assert res.answer.claims == [] and res.stats.critic_rounds == 2


def test_critic_loop_repairs_a_misread(retriever):
    """A researcher that always misreads first, then reads correctly, gets fixed by the critic."""
    from analyst.llm.mock import MockLLM

    llm = MockLLM(p_misread=1.0, p_critic_miss=0, p_critic_false_reject=0)
    q = "What was Lumora Systems's fiscal 2024 revenue?"
    no_critic = MultiAgentPipeline(llm, retriever, MultiAgentConfig(max_critic_rounds=0)).run(q)
    with_critic = MultiAgentPipeline(llm, retriever).run(q)
    assert no_critic.answer.claims[0].value != "2,817.4"
    # After rejection the misread value is excluded, so the researcher must find the true one
    # or abstain; it may not repeat the rejected value.
    assert all(c.value != no_critic.answer.claims[0].value for c in with_critic.answer.claims)


def test_guardrails_quarantine_injection_docs(llm, retriever):
    res = MultiAgentPipeline(llm, retriever).run("What was Lumora Systems's fiscal 2024 revenue?")
    # whether or not an injection doc was retrieved, nothing quoting $9,999.0 may be claimed
    assert all("9,999" not in c.value for c in res.answer.claims)


def test_single_agent_baseline_runs(llm, retriever):
    res = SingleAgentPipeline(llm, retriever).run(Q_COMPARE)
    assert res.stats.llm_calls == 1 and res.answer.summary


def test_budget_exhaustion_degrades_to_abstention(perfect_llm, retriever):
    cfg = MultiAgentConfig(budget=Budget(max_llm_calls=2))
    res = MultiAgentPipeline(perfect_llm, retriever, cfg).run(Q_COMPARE)
    assert any("budget" in e for e in res.answer.guardrail_events)
    assert res.stats.llm_calls == 2


def test_trace_records_every_llm_call(perfect_llm, retriever, tmp_path):
    import json

    pipe = MultiAgentPipeline(perfect_llm, retriever)
    res = pipe.run(Q_COMPARE)
    trace = json.loads(pipe.last_tracer.save(tmp_path).read_text())
    llm_spans = [s for s in trace["spans"] if s["name"].startswith("llm.")]
    assert len(llm_spans) == res.stats.llm_calls
    assert sum(s["attributes"]["input_tokens"] for s in llm_spans) == res.stats.input_tokens


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_invalid_json_retries_then_fails(retriever):
    class Broken:
        model = "claude-opus-5-5"
        calls = 0

        def complete(self, req):
            Broken.calls += 1
            return LLMResponse(text="not json", input_tokens=1, output_tokens=1, model=self.model)

    with pytest.raises(AgentOutputError):
        MultiAgentPipeline(Broken(), retriever).run(Q_COMPARE)
    assert Broken.calls == 2
