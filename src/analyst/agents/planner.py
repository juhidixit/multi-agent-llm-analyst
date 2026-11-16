from __future__ import annotations

from analyst.agents.base import ATTRIBUTE_HELP, Agent
from analyst.types import Attribute, Plan, SubTask

SYSTEM = f"""You are the planning agent in a financial research team.
Decompose the user's question into atomic look-ups. Each look-up is one company and one
attribute from this closed list:
{ATTRIBUTE_HELP}

Rules:
- Use company names exactly as written in the question.
- If the question compares companies on one attribute, set "comparison" to that attribute.
- Do not answer the question. Only plan.

Reply with JSON only:
{{"subtasks": [{{"entity": "...", "attribute": "...", "query": "search query"}}],
  "comparison": "<attribute or null>"}}"""


class Planner(Agent):
    role = "planner"

    def plan(self, question: str) -> Plan:
        with self.tracer.span("agent.planner"):
            raw = self.call(SYSTEM, f"Question: {question}", {"question": question})
            subtasks: list[SubTask] = []
            seen: set[tuple[str, str]] = set()
            valid = {a.value for a in Attribute}
            for i, st in enumerate(raw.get("subtasks", [])):
                key = (st.get("entity", "").strip(), st.get("attribute", ""))
                if not key[0] or key[1] not in valid or key in seen:
                    continue  # drop malformed or duplicate look-ups
                seen.add(key)
                subtasks.append(
                    SubTask(
                        id=f"t{i}",
                        entity=key[0],
                        attribute=Attribute(key[1]),
                        query=st.get("query") or f"{key[0]} {key[1]}",
                    )
                )
            comp = raw.get("comparison")
            return Plan(subtasks=subtasks, comparison=Attribute(comp) if comp in valid else None)
