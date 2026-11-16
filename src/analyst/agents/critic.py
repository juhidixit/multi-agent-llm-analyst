from __future__ import annotations

from analyst.agents.base import UNTRUSTED_NOTE, Agent
from analyst.memory import SharedMemory
from analyst.types import ATTRIBUTE_LABELS, Claim, Verdict

SYSTEM = f"""You are the critic agent in a financial research team. You verify claims.
For each claim, decide whether the CITED passages explicitly support it: same company,
same attribute, same period, same value (ignoring formatting such as commas or "$").
A value that appears in the passage but refers to another year, another role (e.g. a former
CEO), another location (e.g. an R&D site), or another company does NOT support the claim.
When unsupported, give one sentence of actionable feedback.
{UNTRUSTED_NOTE}

Reply with JSON only:
{{"verdicts": [{{"claim_id": "...", "supported": true, "feedback": ""}}]}}"""


class Critic(Agent):
    role = "critic"

    def review(self, claims: list[Claim], memory: SharedMemory, round_: int) -> list[Verdict]:
        if not claims:
            return []
        with self.tracer.span("agent.critic", round=round_, n_claims=len(claims)):
            cited = {d for c in claims for d in c.citations}
            evidence = {d: memory.evidence[d] for d in cited if d in memory.evidence}
            blocks = []
            for c in claims:
                ev = "\n".join(
                    f"  [{d}] {evidence[d].text}"
                    if d in evidence
                    else f"  [{d}] (not in retrieved evidence)"
                    for d in c.citations
                )
                blocks.append(
                    f'claim_id={c.id}: "{c.text}" '
                    f"(entity={c.entity}; attribute={ATTRIBUTE_LABELS[c.attribute]}; "
                    f"value={c.value})\n{ev or '  (no citations)'}"
                )
            raw = self.call(
                SYSTEM,
                "Claims to verify:\n\n" + "\n\n".join(blocks),
                {
                    "round": round_,
                    "claims": [c.model_dump(mode="json") for c in claims],
                    "evidence": {d: p.model_dump() for d, p in evidence.items()},
                },
            )
            by_id = {v.get("claim_id"): v for v in raw.get("verdicts", [])}
            out = []
            for c in claims:
                v = by_id.get(c.id)
                # A claim the critic did not rule on is treated as unsupported (fail closed).
                verdict = Verdict(
                    claim_id=c.id,
                    supported=bool(v and v.get("supported")),
                    feedback=(v or {}).get("feedback", "no verdict returned"),
                )
                memory.write_verdict(verdict, c.subtask_id)
                out.append(verdict)
            return out
