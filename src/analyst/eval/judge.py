"""Score answers against the gold fact table.

The judge is independent of the critic: the critic only sees the passages a claim cites,
while the judge looks up the ground-truth fact table that generated the corpus. A claim is
**hallucinated** when the value it asserts is not the corpus-supported value for that
(entity, attribute): a wrong value, a value for a fact the corpus does not contain, or a
fact about a company that does not exist in the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from analyst.pipelines.synthesis import to_number
from analyst.types import Answer


@dataclass
class GoldFact:
    value: str
    in_corpus: bool


def load_facts(path: Path) -> dict[tuple[str, str], GoldFact]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["entity"].lower(), r["attribute"])] = GoldFact(r["value"], r["in_corpus"])
    return out


def values_match(pred: str, gold: str) -> bool:
    p, g = to_number(pred), to_number(gold)
    if p is not None and g is not None and any(ch.isdigit() for ch in gold):
        return abs(p - g) <= max(0.05, abs(g) * 0.005)
    return pred.strip().lower().rstrip(".") == gold.strip().lower()


@dataclass
class QuestionScore:
    qid: str
    n_claims: int = 0
    n_hallucinated: int = 0
    answerable: int = 0
    answerable_correct: int = 0
    unanswerable: int = 0
    unanswerable_abstained: int = 0
    fully_correct: bool = False
    hallucinations: list[dict] = field(default_factory=list)


def score_answer(
    qid: str, answer: Answer, required: list[dict], facts: dict[tuple[str, str], GoldFact]
) -> QuestionScore:
    s = QuestionScore(qid=qid)
    correct_keys = set()
    for c in answer.claims:
        s.n_claims += 1
        key = (c.entity.lower(), c.attribute.value)
        gold = facts.get(key)
        ok = gold is not None and gold.in_corpus and values_match(c.value, gold.value)
        if ok:
            correct_keys.add(key)
        else:
            s.n_hallucinated += 1
            s.hallucinations.append(
                {
                    "entity": c.entity,
                    "attribute": c.attribute.value,
                    "claimed": c.value,
                    "gold": gold.value if gold and gold.in_corpus else None,
                    "citations": c.citations,
                }
            )
    claimed_keys = {(c.entity.lower(), c.attribute.value) for c in answer.claims}
    for r in required:
        key = (r["entity"].lower(), r["attribute"])
        gold = facts.get(key)
        if gold is not None and gold.in_corpus:
            s.answerable += 1
            s.answerable_correct += key in correct_keys
        else:
            s.unanswerable += 1
            s.unanswerable_abstained += key not in claimed_keys
    s.fully_correct = (
        s.n_hallucinated == 0
        and s.answerable_correct == s.answerable
        and s.unanswerable_abstained == s.unanswerable
    )
    return s
