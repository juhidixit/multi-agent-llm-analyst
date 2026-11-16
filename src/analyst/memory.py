"""Shared memory (blackboard) that the planner, researcher and critic read from and write to.

Every write is an append-only event, so the full history of a run (what was retrieved,
which drafts were rejected and why, how they were revised) can be inspected afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analyst.types import Abstention, Claim, Passage, Plan, Verdict


@dataclass
class MemoryEvent:
    seq: int
    kind: str  # plan | evidence | claim | abstention | verdict | guardrail
    author: str
    subtask_id: str | None
    data: dict[str, Any]


@dataclass
class SharedMemory:
    events: list[MemoryEvent] = field(default_factory=list)
    evidence: dict[str, Passage] = field(default_factory=dict)

    def _write(
        self, kind: str, author: str, data: dict[str, Any], subtask_id: str | None = None
    ) -> None:
        self.events.append(MemoryEvent(len(self.events), kind, author, subtask_id, data))

    # -- writers ------------------------------------------------------------------------
    def write_plan(self, plan: Plan) -> None:
        self._write("plan", "planner", plan.model_dump(mode="json"))

    def write_evidence(self, author: str, subtask_id: str, passages: list[Passage]) -> None:
        for p in passages:
            self.evidence.setdefault(p.doc_id, p)
        self._write("evidence", author, {"doc_ids": [p.doc_id for p in passages]}, subtask_id)

    def write_claim(self, claim: Claim) -> None:
        self._write("claim", "researcher", claim.model_dump(mode="json"), claim.subtask_id)

    def write_abstention(self, a: Abstention) -> None:
        self._write("abstention", "researcher", a.model_dump(mode="json"), a.subtask_id)

    def write_verdict(self, v: Verdict, subtask_id: str) -> None:
        self._write("verdict", "critic", v.model_dump(mode="json"), subtask_id)

    def write_guardrail(self, message: str, subtask_id: str | None = None) -> None:
        self._write("guardrail", "guardrails", {"message": message}, subtask_id)

    # -- readers ------------------------------------------------------------------------
    def read(self, kind: str | None = None, subtask_id: str | None = None) -> list[MemoryEvent]:
        return [
            e
            for e in self.events
            if (kind is None or e.kind == kind)
            and (subtask_id is None or e.subtask_id == subtask_id)
        ]

    def rejected_values(self, subtask_id: str) -> list[str]:
        """Values the critic has already rejected for this subtask (so they aren't retried)."""
        claims = {e.data["id"]: e.data["value"] for e in self.read("claim", subtask_id)}
        return [
            claims[e.data["claim_id"]]
            for e in self.read("verdict", subtask_id)
            if not e.data["supported"] and e.data["claim_id"] in claims
        ]

    def feedback(self, subtask_id: str) -> list[str]:
        return [
            e.data["feedback"]
            for e in self.read("verdict", subtask_id)
            if not e.data["supported"] and e.data["feedback"]
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        return [e.__dict__ for e in self.events]
