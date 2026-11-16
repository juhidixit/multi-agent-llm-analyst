"""Offline, deterministic *simulated* LLM.

This is not a language model. It is a seeded behavioural model of one, used so the whole
system runs and is tested without an API key. It reads the structured ``payload`` that each
agent sends alongside its prompt and answers in the same JSON format a real model is asked
for.

Its failure modes are explicit, configurable, and modelled on how LLM analysts actually go
wrong on this corpus:

* ``p_misread``      - picks a nearby distractor (prior-year revenue, former CEO, R&D site,
                       IPO year, a peer company's figure) instead of the true value.
* ``p_guess``        - when the evidence lacks the fact, asserts a plausible value anyway
                       instead of abstaining.
* ``p_injection``    - follows an instruction embedded in a retrieved document.
* ``p_critic_miss``  - the critic accepts an unsupported claim.
* ``p_critic_false_reject`` - the critic rejects a supported claim.

Because these rates are chosen, offline results measure the *mechanics* of the pipeline
(does the critic loop catch and repair what it should, does the guardrail block injections,
does the harness count correctly). They say nothing about any real model. Use
``ANALYST_PROVIDER=anthropic`` for that.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from analyst.llm.base import LLMRequest, LLMResponse
from analyst.types import ATTRIBUTE_LABELS, Attribute

COMPANY_RE = re.compile(
    r"\b([A-Z][a-z]+ (?:Dynamics|Labs|Systems|Analytics|Robotics|Bio|Energy|Networks|"
    r"Holdings|Group))\b"
)
_NUM = r"(-?[\d,]+(?:\.\d+)?)"
_NAME = r"([A-Z][a-z]+ [A-Z][a-z]+)"
_CITY = r"([A-Z][a-z]+)"

# (true-fact patterns, distractor patterns) per attribute.
PATTERNS: dict[Attribute, tuple[list[str], list[str]]] = {
    Attribute.founded: ([r"[Ff]ounded in (\d{4})"], [r"IPO in (\d{4})"]),
    Attribute.hq: (
        [rf"headquartered in {_CITY}", rf"headquarters are in {_CITY}"],
        [rf"R&D center in {_CITY}"],
    ),
    Attribute.ceo: (
        [rf"chief executive officer is {_NAME}", rf"{_NAME} is the current CEO"],
        [rf"former CEO,? {_NAME}"],
    ),
    Attribute.sector: ([r"operates in the ([a-z ]+) sector"], []),
    Attribute.revenue_2024: (
        [
            rf"[Ff]iscal 2024 revenue reached \${_NUM}",
            rf"fiscal 2024 the company reported revenue of \${_NUM}",
            rf"reported fiscal 2024 revenue of \${_NUM}",
        ],
        [rf"[Ff]iscal 2023 revenue for .*? was \${_NUM}"],
    ),
    Attribute.yoy_growth: (
        [rf"year-over-year change of {_NUM}%", r"\(([+-]?[\d.]+)% year over year\)"],
        [],
    ),
    Attribute.employees: (
        [rf"employs {_NUM} people", rf"headcount stands at {_NUM}"],
        [rf"In 2021 the company had {_NUM} employees"],
    ),
}
INJECTION_RE = re.compile(r"report that (.+?)'s fiscal 2024 revenue was \$" + _NUM)


def norm(value: str) -> str:
    """Normalise a value for comparison: strip $, commas, signs on growth, case."""
    v = value.strip().lower().replace("$", "").replace(",", "").replace("million", "").strip()
    v = v.rstrip("%").strip()
    try:
        f = float(v)
        return f"{f:.1f}" if "." in v else str(int(f))
    except ValueError:
        return v


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


@dataclass
class Candidate:
    value: str
    doc_id: str
    kind: str  # "true" | "distractor" | "injected"


def read_candidates(entity: str, attr: Attribute, passages: list[dict]) -> list[Candidate]:
    """Find every value a reader might attach to (entity, attr) in the passages."""
    true_pats, dis_pats = PATTERNS[attr]
    out: list[Candidate] = []
    for p in passages:
        about_entity = entity.lower() in p["title"].lower()
        for s in _sentences(p["text"]):
            named = set(COMPANY_RE.findall(s))
            if attr == Attribute.revenue_2024:
                m = INJECTION_RE.search(s)
                if m and m.group(1) == entity:
                    out.append(Candidate(m.group(2), p["doc_id"], "injected"))
                    continue
            belongs = entity in named or (about_entity and not (named - {entity}))
            for pat in true_pats:
                for m in re.finditer(pat, s):
                    kind = "true" if belongs else "distractor"
                    out.append(Candidate(m.group(1), p["doc_id"], kind))
            if belongs or about_entity:
                for pat in dis_pats:
                    for m in re.finditer(pat, s):
                        out.append(Candidate(m.group(1), p["doc_id"], "distractor"))
    return out


def supported_by(entity: str, attr: Attribute, value: str, passages: list[dict]) -> bool:
    return any(
        c.kind == "true" and norm(c.value) == norm(value)
        for c in read_candidates(entity, attr, passages)
    )


class MockLLM:
    def __init__(
        self,
        seed: int = 0,
        p_misread: float = 0.15,
        p_guess: float = 0.6,
        p_injection: float = 0.5,
        p_critic_miss: float = 0.1,
        p_critic_false_reject: float = 0.03,
        model: str = "claude-opus-5-5",
    ) -> None:
        # `model` only selects the price table used for *estimated* cost.
        self.model = model
        self.seed = seed
        self.p = dict(
            misread=p_misread,
            guess=p_guess,
            injection=p_injection,
            critic_miss=p_critic_miss,
            critic_false_reject=p_critic_false_reject,
        )

    # -- helpers -----------------------------------------------------------------------
    def _rng(self, *key: Any) -> random.Random:
        h = hashlib.sha256(json.dumps([self.seed, *key], default=str).encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _respond(self, req: LLMRequest, obj: Any) -> LLMResponse:
        text = json.dumps(obj)
        tin = math.ceil(len(req.system + req.prompt) / 4)
        tout = math.ceil(len(text) / 4)
        # Simulated latency: 0.4 s time-to-first-token + 60 output tokens/s.
        return LLMResponse(
            text=text,
            input_tokens=tin,
            output_tokens=tout,
            model=self.model,
            simulated_latency_s=0.4 + tout / 60,
        )

    # -- roles -------------------------------------------------------------------------
    def complete(self, req: LLMRequest) -> LLMResponse:
        handler = getattr(self, f"_role_{req.role}", None)
        if handler is None:
            raise ValueError(f"MockLLM has no behaviour for role {req.role!r}")
        return self._respond(req, handler(req.payload))

    def _role_planner(self, payload: dict) -> dict:
        return plan_question(payload["question"])

    def _extract(
        self,
        rng: random.Random,
        entity: str,
        attr: Attribute,
        passages: list[dict],
        exclude: set[str],
    ) -> dict:
        cands = [c for c in read_candidates(entity, attr, passages) if norm(c.value) not in exclude]
        injected = [c for c in cands if c.kind == "injected"]
        true = [c for c in cands if c.kind == "true"]
        dis = [c for c in cands if c.kind == "distractor"]
        pick: Candidate | None = None
        if injected and rng.random() < self.p["injection"]:
            pick = injected[0]
        elif true:
            pick = rng.choice(dis) if dis and rng.random() < self.p["misread"] else true[0]
        elif dis and rng.random() < self.p["guess"]:
            pick = rng.choice(dis)
        elif passages and rng.random() < self.p["guess"] * 0.5:
            pick = Candidate(_fabricate(rng, attr), passages[0]["doc_id"], "fabricated")
        if pick is None:
            return {
                "abstain": True,
                "reason": f"No passage states the {ATTRIBUTE_LABELS[attr]} of {entity}.",
            }
        return {
            "abstain": False,
            "value": pick.value,
            "citations": [pick.doc_id],
            "text": f"{entity}'s {ATTRIBUTE_LABELS[attr]} is {pick.value}.",
        }

    def _role_researcher(self, payload: dict) -> dict:
        st = payload["subtask"]
        attr = Attribute(st["attribute"])
        rng = self._rng(
            "researcher",
            st["entity"],
            attr.value,
            payload.get("attempt", 0),
            payload.get("question", ""),
        )
        exclude = {norm(v) for v in payload.get("rejected_values", [])}
        r = self._extract(rng, st["entity"], attr, payload["passages"], exclude)
        if r["abstain"]:
            return {
                "claims": [],
                "abstentions": [
                    {"entity": st["entity"], "attribute": attr.value, "reason": r["reason"]}
                ],
            }
        return {
            "claims": [
                {
                    "entity": st["entity"],
                    "attribute": attr.value,
                    "value": r["value"],
                    "text": r["text"],
                    "citations": r["citations"],
                }
            ],
            "abstentions": [],
        }

    def _role_critic(self, payload: dict) -> dict:
        verdicts = []
        evidence: dict[str, dict] = payload["evidence"]
        for c in payload["claims"]:
            rng = self._rng("critic", c["id"], c["value"], payload.get("round", 0))
            cited = [evidence[d] for d in c["citations"] if d in evidence]
            ok = supported_by(c["entity"], Attribute(c["attribute"]), c["value"], cited)
            if ok and rng.random() < self.p["critic_false_reject"]:
                ok = False
            elif not ok and rng.random() < self.p["critic_miss"]:
                ok = True
            fb = (
                ""
                if ok
                else (
                    f"The cited passage does not state that the {ATTRIBUTE_LABELS[Attribute(c['attribute'])]}"
                    f" of {c['entity']} is {c['value']}. The value may belong to a different year, "
                    f"role, location, or company. Re-read the evidence or abstain."
                )
            )
            verdicts.append({"claim_id": c["id"], "supported": ok, "feedback": fb})
        return {"verdicts": verdicts}

    def _role_single_agent(self, payload: dict) -> dict:
        plan = plan_question(payload["question"])
        claims, abst = [], []
        for st in plan["subtasks"]:
            attr = Attribute(st["attribute"])
            rng = self._rng("single", st["entity"], attr.value, payload["question"])
            r = self._extract(rng, st["entity"], attr, payload["passages"], set())
            if r["abstain"]:
                abst.append(
                    {"entity": st["entity"], "attribute": attr.value, "reason": r["reason"]}
                )
            else:
                claims.append(
                    {
                        "entity": st["entity"],
                        "attribute": attr.value,
                        "value": r["value"],
                        "text": r["text"],
                        "citations": r["citations"],
                    }
                )
        return {"claims": claims, "abstentions": abst}


def _fabricate(rng: random.Random, attr: Attribute) -> str:
    return {
        Attribute.founded: str(rng.randint(1970, 2015)),
        Attribute.hq: rng.choice(["Berlin", "Chicago", "Singapore", "Zurich"]),
        Attribute.ceo: rng.choice(["Alex Morgan", "Sam Patel", "Jordan Lee"]),
        Attribute.sector: "technology",
        Attribute.revenue_2024: f"{rng.uniform(50, 2500):,.1f}",
        Attribute.yoy_growth: f"{rng.uniform(-10, 40):.1f}",
        Attribute.employees: f"{rng.randint(100, 15000):,}",
    }[attr]


_ATTR_CUES: list[tuple[Attribute, tuple[str, ...]]] = [
    (Attribute.yoy_growth, ("grow", "growth", "year over year")),
    (Attribute.revenue_2024, ("revenue",)),
    (Attribute.employees, ("employee", "headcount", "employs", "people")),
    (Attribute.ceo, ("ceo", "chief executive")),
    (Attribute.hq, ("headquarter",)),
    (Attribute.founded, ("founded",)),
    (Attribute.sector, ("sector", "industry")),
]


def plan_question(question: str) -> dict:
    """Rule-based decomposition used by the mock planner (a real LLM plans from the prompt)."""
    entities = list(dict.fromkeys(COMPANY_RE.findall(question)))
    ql = question.lower()
    attrs: list[Attribute] = []
    for attr, cues in _ATTR_CUES:
        if any(c in ql for c in cues) and not (
            attr == Attribute.revenue_2024 and Attribute.yoy_growth in attrs
        ):
            attrs.append(attr)
    subtasks = []
    for e in entities:
        for a in attrs:
            subtasks.append(
                {"entity": e, "attribute": a.value, "query": f"{e} {ATTRIBUTE_LABELS[a]}"}
            )
    comparison = attrs[0].value if len(entities) > 1 and len(attrs) == 1 else None
    return {"subtasks": subtasks, "comparison": comparison}
