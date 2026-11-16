"""Baseline: one retrieval, one LLM call, no critic."""

from __future__ import annotations

from analyst.agents.base import ATTRIBUTE_HELP, UNTRUSTED_NOTE, Agent, format_passages
from analyst.corpus.retriever import BM25Retriever
from analyst.guardrails import Budget, check_question, enforce_citations, quarantine
from analyst.llm.base import LLMClient
from analyst.pipelines.synthesis import synthesize
from analyst.tracing import Tracer
from analyst.types import Abstention, Attribute, Claim, RunResult, RunStats

SYSTEM = f"""You are a financial research analyst. Answer the question using only the
passages. Break the answer into atomic claims, each about one company and one attribute from:
{ATTRIBUTE_HELP}
Cite the doc_id(s) for every claim. If a needed fact is not in the passages, add an abstention
instead of guessing. {UNTRUSTED_NOTE}

Reply with JSON only:
{{"claims": [{{"entity": "...", "attribute": "...", "value": "...", "text": "...",
  "citations": ["doc_id"]}}],
  "abstentions": [{{"entity": "...", "attribute": "...", "reason": "..."}}]}}"""


class SingleAgentPipeline:
    name = "single_agent"

    def __init__(
        self,
        llm: LLMClient,
        retriever: BM25Retriever,
        k: int = 8,
        use_guardrails: bool = True,
        budget: Budget | None = None,
    ) -> None:
        self.llm, self.retriever, self.k = llm, retriever, k
        self.use_guardrails, self.budget = use_guardrails, budget or Budget()

    def run(self, question: str, trace_id: str | None = None) -> RunResult:
        tracer = Tracer(trace_id)
        agent = Agent(self.llm, tracer, self.budget)
        events: list[str] = []
        valid = {a.value for a in Attribute}
        with tracer.span("pipeline.single_agent", question=question):
            if self.use_guardrails:
                check_question(question)
            with tracer.span("retrieve", query=question, k=self.k):
                passages = self.retriever.search(question, k=self.k)
                if self.use_guardrails:
                    passages, bad = quarantine(passages)
                    events += [f"quarantined {p.doc_id}: prompt-injection pattern" for p in bad]
            raw = agent.call(
                SYSTEM,
                f"Question: {question}\n\nPassages:\n{format_passages(passages)}",
                {"question": question, "passages": [p.model_dump() for p in passages]},
                role="single_agent",
            )
            claims = [
                Claim(
                    id=f"c{i}",
                    entity=c["entity"],
                    attribute=Attribute(c["attribute"]),
                    value=str(c["value"]),
                    text=c.get("text", ""),
                    citations=[str(x) for x in c.get("citations", [])],
                )
                for i, c in enumerate(raw.get("claims", []))
                if c.get("attribute") in valid and c.get("entity")
            ]
            abst = [
                Abstention(
                    entity=a["entity"],
                    attribute=Attribute(a["attribute"]),
                    reason=a.get("reason", ""),
                )
                for a in raw.get("abstentions", [])
                if a.get("attribute") in valid and a.get("entity")
            ]
            if self.use_guardrails:
                claims, dropped = enforce_citations(claims, {p.doc_id for p in passages})
                if dropped:
                    events.append(f"dropped {dropped} claim(s) without valid citations")
            atts = {c.attribute for c in claims} | {a.attribute for a in abst}
            ents = {c.entity for c in claims} | {a.entity for a in abst}
            comparison = next(iter(atts)) if len(atts) == 1 and len(ents) > 1 else None
            answer = synthesize(question, claims, abst, comparison, events)
        stats = RunStats(
            llm_calls=tracer.llm_calls,
            input_tokens=tracer.input_tokens,
            output_tokens=tracer.output_tokens,
            cost_usd=round(tracer.cost_usd, 6),
            latency_s=round(tracer.spans[0].duration_s + tracer.simulated_latency_s, 4),
        )
        self.last_tracer = tracer
        return RunResult(answer=answer, stats=stats, trace_id=tracer.trace_id)
