"""HTTP API. `uvicorn analyst.api:app` or `analyst serve`."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analyst.config import data_dir
from analyst.corpus.retriever import BM25Retriever
from analyst.eval.harness import SYSTEMS
from analyst.guardrails import MAX_QUESTION_CHARS, BudgetExceeded, GuardrailViolation
from analyst.llm import make_client
from analyst.types import RunResult

app = FastAPI(title="Multi-Agent LLM Analyst", version="0.1.0")


class AnalyzeRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS)
    pipeline: Literal["multi_agent", "single_agent"] = "multi_agent"


@lru_cache(maxsize=1)
def _retriever() -> BM25Retriever:
    return BM25Retriever.from_jsonl(data_dir() / "corpus.jsonl")


@lru_cache(maxsize=1)
def _llm():
    return make_client()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "provider": type(_llm()).__name__}


@app.post("/v1/analyze", response_model=RunResult)
def analyze(req: AnalyzeRequest) -> RunResult:
    pipe = SYSTEMS[req.pipeline](_llm(), _retriever())
    try:
        return pipe.run(req.question)
    except GuardrailViolation as exc:
        raise HTTPException(status_code=422, detail=f"guardrail: {exc}") from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
