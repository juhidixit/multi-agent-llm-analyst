"""Deterministic answer synthesis. Arithmetic and comparisons are done in code, never by the model."""

from __future__ import annotations

import re

from analyst.guardrails import redact_pii
from analyst.types import ATTRIBUTE_LABELS, Abstention, Answer, Attribute, Claim

NUMERIC = {Attribute.revenue_2024, Attribute.employees, Attribute.founded, Attribute.yoy_growth}


def to_number(value: str) -> float | None:
    m = re.search(r"-?[\d,]*\.?\d+", value.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def synthesize(
    question: str,
    claims: list[Claim],
    abstentions: list[Abstention],
    comparison: Attribute | None,
    guardrail_events: list[str],
) -> Answer:
    parts = [f"{c.text} [{', '.join(c.citations)}]" for c in claims]
    if comparison and comparison in NUMERIC:
        vals: list[tuple[str, float]] = []
        for c in claims:
            n = to_number(c.value)
            if c.attribute == comparison and n is not None:
                vals.append((c.entity, n))
        if len(vals) >= 2 and not any(a.attribute == comparison for a in abstentions):
            (e1, v1), (e2, v2) = sorted(vals, key=lambda x: -x[1])[:2]
            label = ATTRIBUTE_LABELS[comparison]
            if comparison == Attribute.founded:
                (e1, v1), (e2, v2) = (e2, v2), (e1, v1)
                parts.append(f"{e1} was founded earlier, by {int(v2 - v1)} years.")
            else:
                parts.append(f"{e1} is higher on {label} by {v1 - v2:,.1f} (computed).")
    for a in abstentions:
        parts.append(f"Not stated in the sources: {ATTRIBUTE_LABELS[a.attribute]} of {a.entity}.")
    summary = redact_pii(" ".join(parts) if parts else "I could not find an answer in the sources.")
    return Answer(
        question=question,
        summary=summary,
        claims=claims,
        abstentions=abstentions,
        guardrail_events=guardrail_events,
    )
