"""Planner -> researcher(s) -> critic loop, coordinated through shared memory."""

from __future__ import annotations

from dataclasses import dataclass

from analyst.agents import Critic, Planner, Researcher
from analyst.corpus.retriever import BM25Retriever
from analyst.guardrails import Budget, BudgetExceeded, check_question, enforce_citations
from analyst.llm.base import LLMClient
from analyst.memory import SharedMemory
from analyst.pipelines.synthesis import synthesize
from analyst.tracing import Tracer
from analyst.types import Abstention, Claim, RunResult, RunStats


@dataclass
class MultiAgentConfig:
    max_critic_rounds: int = 2  # 0 disables the critic (ablation)
    k: int = 4
    use_guardrails: bool = True
    budget: Budget | None = None


class MultiAgentPipeline:
    name = "multi_agent"

    def __init__(
        self, llm: LLMClient, retriever: BM25Retriever, config: MultiAgentConfig | None = None
    ) -> None:
        self.llm, self.retriever = llm, retriever
        self.cfg = config or MultiAgentConfig()

    def run(self, question: str, trace_id: str | None = None) -> RunResult:
        tracer = Tracer(trace_id)
        self.last_tracer = tracer  # set first, so spend is visible even if the run fails
        memory = SharedMemory()
        budget = self.cfg.budget or Budget()
        planner = Planner(self.llm, tracer, budget)
        researcher = Researcher(
            self.llm,
            tracer,
            budget,
            retriever=self.retriever,
            k=self.cfg.k,
            use_guardrails=self.cfg.use_guardrails,
        )
        critic = Critic(self.llm, tracer, budget)
        events: list[str] = []
        rounds = 0

        with tracer.span("pipeline.multi_agent", question=question) as root:
            if self.cfg.use_guardrails:
                check_question(question)
            plan = planner.plan(question)
            memory.write_plan(plan)
            results: dict[str, Claim | Abstention] = {}
            try:
                for st in plan.subtasks:
                    results[st.id] = researcher.research(question, st, memory)

                for rnd in range(1, self.cfg.max_critic_rounds + 1):
                    pending = [r for r in results.values() if isinstance(r, Claim)]
                    verdicts = critic.review(pending, memory, rnd)
                    rounds = rnd
                    rejected = [v.claim_id for v in verdicts if not v.supported]
                    if not rejected:
                        break
                    final_round = rnd == self.cfg.max_critic_rounds
                    for st in plan.subtasks:
                        r = results[st.id]
                        if isinstance(r, Claim) and r.id in rejected:
                            if final_round:
                                # Never ship a claim the critic rejected: abstain instead.
                                results[st.id] = Abstention(
                                    subtask_id=st.id,
                                    entity=st.entity,
                                    attribute=st.attribute,
                                    reason="critic could not verify any candidate value",
                                )
                            else:
                                results[st.id] = researcher.research(
                                    question, st, memory, attempt=rnd
                                )
            except BudgetExceeded as exc:
                events.append(str(exc))
                for st in plan.subtasks:
                    if st.id not in results:
                        results[st.id] = Abstention(
                            subtask_id=st.id,
                            entity=st.entity,
                            attribute=st.attribute,
                            reason=str(exc),
                        )

            claims = [r for r in results.values() if isinstance(r, Claim)]
            abstentions = [r for r in results.values() if isinstance(r, Abstention)]
            if self.cfg.use_guardrails:
                claims, dropped = enforce_citations(claims, set(memory.evidence))
                if dropped:
                    events.append(f"dropped {dropped} claim(s) without valid citations")
            events += [e.data["message"] for e in memory.read("guardrail")]
            answer = synthesize(question, claims, abstentions, plan.comparison, events)
            root.attributes["memory_events"] = len(memory.events)

        wall = tracer.spans[0].duration_s
        stats = RunStats(
            llm_calls=tracer.llm_calls,
            input_tokens=tracer.input_tokens,
            output_tokens=tracer.output_tokens,
            cost_usd=round(tracer.cost_usd, 6),
            latency_s=round(wall + tracer.simulated_latency_s, 4),
            critic_rounds=rounds,
        )
        self.last_tracer, self.last_memory = tracer, memory
        return RunResult(answer=answer, stats=stats, trace_id=tracer.trace_id)
