"""Typed data contracts shared by agents, pipelines, and the evaluation harness."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Attribute(str, Enum):
    """Closed set of facts the analyst can assert. Keeping this closed makes claims checkable."""

    founded = "founded"
    hq = "hq"
    ceo = "ceo"
    sector = "sector"
    revenue_2024 = "revenue_2024"
    yoy_growth = "yoy_growth"
    employees = "employees"


ATTRIBUTE_LABELS: dict[Attribute, str] = {
    Attribute.founded: "founding year",
    Attribute.hq: "headquarters city",
    Attribute.ceo: "current CEO",
    Attribute.sector: "sector",
    Attribute.revenue_2024: "fiscal 2024 revenue (USD millions)",
    Attribute.yoy_growth: "fiscal 2024 year-over-year revenue growth (%)",
    Attribute.employees: "current employee headcount",
}


class Passage(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    quarantined: bool = False


class SubTask(BaseModel):
    id: str
    entity: str
    attribute: Attribute
    query: str


class Plan(BaseModel):
    subtasks: list[SubTask]
    comparison: Attribute | None = Field(
        default=None, description="If the question compares entities on one attribute."
    )


class Claim(BaseModel):
    id: str = ""
    subtask_id: str = ""
    entity: str
    attribute: Attribute
    value: str
    text: str
    citations: list[str] = Field(default_factory=list)


class Abstention(BaseModel):
    subtask_id: str = ""
    entity: str
    attribute: Attribute
    reason: str


class Verdict(BaseModel):
    claim_id: str
    supported: bool
    feedback: str = ""


class Answer(BaseModel):
    question: str
    summary: str
    claims: list[Claim]
    abstentions: list[Abstention] = Field(default_factory=list)
    guardrail_events: list[str] = Field(default_factory=list)


class RunStats(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    critic_rounds: int = 0


class RunResult(BaseModel):
    answer: Answer
    stats: RunStats
    trace_id: str
