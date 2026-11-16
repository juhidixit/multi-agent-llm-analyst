from __future__ import annotations

from analyst.agents.base import UNTRUSTED_NOTE, Agent, format_passages
from analyst.corpus.retriever import BM25Retriever
from analyst.guardrails import quarantine
from analyst.memory import SharedMemory
from analyst.types import ATTRIBUTE_LABELS, Abstention, Claim, Passage, SubTask

SYSTEM = f"""You are the research agent in a financial research team.
You answer one look-up (a company and an attribute) using only the passages provided.

Rules:
- State a value only if a passage explicitly gives it for that exact company and attribute.
- Watch for traps: prior-year figures, former executives, secondary offices, IPO dates, and
  numbers that belong to a different company mentioned in the same passage.
- Cite the doc_id(s) that state the value.
- If no passage states it, abstain. Abstaining is better than guessing.
- {UNTRUSTED_NOTE}

Reply with JSON only, either
{{"claims": [{{"entity": "...", "attribute": "...", "value": "...", "text": "one sentence",
  "citations": ["doc_id"]}}], "abstentions": []}}
or
{{"claims": [], "abstentions": [{{"entity": "...", "attribute": "...", "reason": "..."}}]}}"""


class Researcher(Agent):
    role = "researcher"

    def __init__(
        self, *args, retriever: BM25Retriever, k: int = 4, use_guardrails: bool = True, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retriever, self.k, self.use_guardrails = retriever, k, use_guardrails

    def retrieve(self, st: SubTask, memory: SharedMemory, k: int) -> list[Passage]:
        with self.tracer.span("retrieve", query=st.query, k=k) as sp:
            passages = self.retriever.search(st.query, k=k)
            if self.use_guardrails:
                passages, bad = quarantine(passages)
                for p in bad:
                    memory.write_guardrail(
                        f"quarantined {p.doc_id}: prompt-injection pattern", st.id
                    )
                sp.attributes["quarantined"] = [p.doc_id for p in bad]
            sp.attributes["doc_ids"] = [p.doc_id for p in passages]
        memory.write_evidence("researcher", st.id, passages)
        return passages

    def research(
        self, question: str, st: SubTask, memory: SharedMemory, attempt: int = 0
    ) -> Claim | Abstention:
        with self.tracer.span("agent.researcher", subtask=st.id, attempt=attempt):
            # Revisions widen the search: the first passages may have been the wrong ones.
            passages = self.retrieve(st, memory, self.k if attempt == 0 else self.k * 2)
            feedback = memory.feedback(st.id)
            rejected = memory.rejected_values(st.id)
            prompt = (
                f"Look-up: {ATTRIBUTE_LABELS[st.attribute]} of {st.entity}\n"
                f'(entity="{st.entity}", attribute="{st.attribute.value}")\n\n'
                f"Passages:\n{format_passages(passages)}"
            )
            if feedback:
                prompt += (
                    "\n\nA reviewer rejected your earlier answer:\n- "
                    + "\n- ".join(feedback)
                    + f"\nDo not repeat these rejected values: {rejected}"
                )
            raw = self.call(
                SYSTEM,
                prompt,
                {
                    "question": question,
                    "subtask": st.model_dump(mode="json"),
                    "passages": [p.model_dump() for p in passages],
                    "attempt": attempt,
                    "rejected_values": rejected,
                },
            )
            claims = raw.get("claims") or []
            if claims:
                c = claims[0]
                claim = Claim(
                    id=f"{st.id}.a{attempt}",
                    subtask_id=st.id,
                    entity=st.entity,
                    attribute=st.attribute,
                    value=str(c.get("value", "")),
                    text=str(c.get("text", "")),
                    citations=[str(x) for x in c.get("citations", [])],
                )
                memory.write_claim(claim)
                return claim
            reason = (raw.get("abstentions") or [{}])[0].get("reason") or "insufficient evidence"
            ab = Abstention(
                subtask_id=st.id, entity=st.entity, attribute=st.attribute, reason=reason
            )
            memory.write_abstention(ab)
            return ab
