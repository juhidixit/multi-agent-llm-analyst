"""Lightweight span tracer with token/cost/latency accounting.

Each pipeline run produces one trace: a tree of spans (agent steps, LLM calls, retrievals,
guardrail checks). Traces are plain JSON so they can be diffed, grepped, or shipped to any
backend. The span model mirrors OpenTelemetry (trace_id / span_id / parent_id / attributes)
so an OTel exporter is a thin adapter.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return (self.end or time.perf_counter()) - self.start


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        # Aggregates updated by LLM calls.
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.simulated_latency_s = 0.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        sp = Span(
            name=name,
            span_id=uuid.uuid4().hex[:12],
            parent_id=parent,
            start=time.perf_counter(),
            attributes=dict(attrs),
        )
        self.spans.append(sp)
        self._stack.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.attributes["error"] = repr(exc)
            raise
        finally:
            sp.end = time.perf_counter()
            self._stack.pop()

    def record_llm(
        self,
        *,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        simulated_latency_s: float = 0.0,
    ) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd
        self.simulated_latency_s += simulated_latency_s
        if self._stack:
            a = self._stack[-1].attributes
            a.update(
                role=role,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost_usd, 6),
            )
            if simulated_latency_s:
                a["simulated_latency_s"] = round(simulated_latency_s, 3)

    def to_dict(self) -> dict[str, Any]:
        t0 = self.spans[0].start if self.spans else 0.0
        return {
            "trace_id": self.trace_id,
            "totals": {
                "llm_calls": self.llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
            },
            "spans": [
                {
                    **{k: v for k, v in asdict(s).items() if k not in ("start", "end")},
                    "offset_ms": round((s.start - t0) * 1000, 2),
                    "duration_ms": round(s.duration_s * 1000, 2),
                }
                for s in self.spans
            ],
        }

    def save(self, directory: str | Path) -> Path:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        out = p / f"{self.trace_id}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return out


def render_tree(trace: dict[str, Any]) -> str:
    """Pretty-print a saved trace as an indented tree (used by `analyst trace`)."""
    children: dict[str | None, list[dict[str, Any]]] = {}
    for s in trace["spans"]:
        children.setdefault(s["parent_id"], []).append(s)
    lines: list[str] = []

    def walk(pid: str | None, depth: int) -> None:
        for s in children.get(pid, []):
            a = s["attributes"]
            extra = ""
            if "input_tokens" in a:
                extra = f"  [{a['input_tokens']}→{a['output_tokens']} tok, ${a['cost_usd']}]"
            lines.append(f"{'  ' * depth}- {s['name']} ({s['duration_ms']} ms){extra}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    t = trace["totals"]
    lines.append(
        f"totals: {t['llm_calls']} LLM calls, {t['input_tokens']} in / "
        f"{t['output_tokens']} out tokens, ${t['cost_usd']}"
    )
    return "\n".join(lines)
