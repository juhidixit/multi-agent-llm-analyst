"""Input, retrieval, and output guardrails plus a hard budget limit."""

from __future__ import annotations

import re
from dataclasses import dataclass

from analyst.types import Claim, Passage

MAX_QUESTION_CHARS = 1000

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (the |all )?(previous|prior|above|system)",
        r"\bsystem (notice|prompt|override)\b",
        r"do not (mention|reveal) this",
        r"you are now",
    ]
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}(?!\d)")


class GuardrailViolation(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def check_question(question: str) -> None:
    if not question.strip():
        raise GuardrailViolation("empty question")
    if len(question) > MAX_QUESTION_CHARS:
        raise GuardrailViolation(f"question longer than {MAX_QUESTION_CHARS} characters")
    if looks_like_injection(question):
        raise GuardrailViolation("question contains a prompt-injection pattern")


def quarantine(passages: list[Passage]) -> tuple[list[Passage], list[Passage]]:
    """Split retrieved passages into (clean, quarantined). Quarantined text never reaches a model."""
    clean, bad = [], []
    for p in passages:
        if looks_like_injection(p.text):
            bad.append(p.model_copy(update={"quarantined": True}))
        else:
            clean.append(p)
    return clean, bad


def redact_pii(text: str) -> str:
    return PHONE.sub("[REDACTED_PHONE]", EMAIL.sub("[REDACTED_EMAIL]", text))


def enforce_citations(claims: list[Claim], allowed_doc_ids: set[str]) -> tuple[list[Claim], int]:
    """Drop claims with no citation or citing a document that was never retrieved."""
    kept = [c for c in claims if c.citations and set(c.citations) <= allowed_doc_ids]
    return kept, len(claims) - len(kept)


@dataclass
class Budget:
    max_llm_calls: int = 40
    max_cost_usd: float = 0.50

    def check(self, llm_calls: int, cost_usd: float) -> None:
        if llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_llm_calls} reached")
        if cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost budget of ${self.max_cost_usd:.2f} reached")
