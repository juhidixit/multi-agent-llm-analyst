# Multi-Agent LLM Analyst

[![ci](https://github.com/juhidixit/multi-agent-llm-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/juhidixit/multi-agent-llm-analyst/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A research-analyst system where a **planner**, a **researcher** and a **critic** agent work
through a **shared memory** to answer questions over a document corpus with cited, checkable
claims, compared against a **single-agent baseline** on a benchmark built to measure
**hallucinated claims**.

It's built to run in production, not only as a notebook demo. Every run is **traced** (span
tree with tokens, cost and latency per LLM call), protected by **guardrails** (prompt-injection
quarantine, citation enforcement, PII redaction, hard per-request budget), exposed as an
**HTTP API**, containerised, tested in **CI**, and deployable to **AWS** with the included
Terraform.

> **Runs with no API key.** The default provider is an offline, seeded *simulated* LLM, so
> `git clone && make install && make test && make eval` works anywhere. Switch to Claude with
> `ANALYST_PROVIDER=anthropic`.

---

## Architecture

```mermaid
flowchart LR
    Q([Question]) --> G1{{Input guardrail}}
    G1 --> P[Planner agent]
    P -- "plan: (company, attribute) look-ups" --> M[(Shared memory<br/>append-only event log)]
    M --> R[Researcher agent]
    R -- query --> BM25[(BM25 retriever)]
    BM25 --> G2{{Injection quarantine}}
    G2 -- clean passages --> R
    R -- "draft claims + citations" --> M
    M --> C[Critic agent]
    C -- "verdicts + feedback" --> M
    C -. "rejected → revise (wider search, rejected values excluded), ≤2 rounds" .-> R
    M --> S[Synthesiser<br/>arithmetic in code]
    S --> G3{{Citation check · PII redaction}}
    G3 --> A([Answer with cited claims<br/>and explicit abstentions])

    T[[Tracer: spans · tokens · $ · latency]] -.- P & R & C
    B[[Budget guard: max calls / $ per request]] -.- P & R & C
```

| Component | What it does | Code |
|---|---|---|
| Planner | Breaks the question into atomic look-ups (company × attribute) from a closed schema. | [`agents/planner.py`](src/analyst/agents/planner.py) |
| Researcher | Retrieves evidence per look-up and drafts one cited claim, or abstains. On revision it widens retrieval and sees the critic's feedback and the values already rejected. | [`agents/researcher.py`](src/analyst/agents/researcher.py) |
| Critic | Checks each claim against **only the passages it cites**. Anything it doesn't explicitly rule on counts as unsupported. | [`agents/critic.py`](src/analyst/agents/critic.py) |
| Shared memory | Append-only blackboard of plan, evidence, claims, verdicts and guardrail events. Agents coordinate through it, and it doubles as an audit log. | [`memory.py`](src/analyst/memory.py) |
| Synthesiser | Builds the final answer. Comparisons and differences are computed in Python, never by the model. | [`pipelines/synthesis.py`](src/analyst/pipelines/synthesis.py) |
| Guardrails | Input injection check, retrieved-passage quarantine, citation enforcement, PII redaction, per-request call/cost budget. | [`guardrails.py`](src/analyst/guardrails.py) |
| Tracing | OpenTelemetry-shaped span tree saved as JSON. `analyst trace <file>` pretty-prints it. | [`tracing.py`](src/analyst/tracing.py) |
| Baseline | One retrieval, one LLM call, same guardrails, no critic. | [`pipelines/single_agent.py`](src/analyst/pipelines/single_agent.py) |

### Design decisions

* **Claims are structured, not free text.** Each claim is `(entity, attribute, value, citations)`.
  This is what makes hallucination *measurable*: a claim can be checked mechanically against
  ground truth instead of graded by vibes.
* **The critic fails closed.** If a claim is still rejected after the last revision round, the
  pipeline abstains for that look-up instead of shipping it. Saying "not in the sources" is a
  valid answer and is scored as one.
* **The critic and the judge are independent.** The critic sees only the passages a claim
  cites. The evaluation judge looks up the gold fact table that generated the corpus. So the
  critic can't grade its own homework.
* **Retrieved text is untrusted.** Passages that match injection patterns are quarantined
  before any model sees them, and every prompt tells the model to treat passages as data.
* **Arithmetic is code.** "By how much?" is answered by subtracting two verified numbers in
  Python.

---

## Benchmark

A hallucination benchmark needs ground truth that a model can't already know, so the corpus
is **synthetic and fictional**: 64 companies, 198 documents, generated deterministically by
[`corpus/generate.py`](src/analyst/corpus/generate.py). It is built to be *tempting*:

* **Distractors next to every fact:** fiscal 2023 vs 2024 revenue, former vs current CEO,
  R&D site vs HQ, IPO year vs founding year, a peer company's figures in the same news item.
* **Missing facts:** about a quarter of companies have one fact absent from the corpus, so the
  correct behaviour is to abstain. Questions about companies that don't exist are included too.
* **Prompt injection:** six "investor forum" documents instruct the model to report a fake
  revenue figure.

| Split | Questions | Purpose |
|---|---|---|
| `dev` | 60 | Development and tuning (16 companies). |
| `test` | 200 | **Held out.** Disjoint companies from `dev` (enforced by a test). Single-fact, comparison, multi-attribute and unknown-company questions. |
| `adversarial` | 18 | Revenue questions whose retrieval surfaces the injection documents. |

**Metrics** ([`eval/judge.py`](src/analyst/eval/judge.py)):

* **Hallucinated-claim rate** = claims whose value is not the corpus-supported value (wrong
  value, a value for a fact the corpus doesn't contain, or a fact about a non-existent company)
  ÷ all claims.
* **Fact recall:** answerable required facts stated correctly.
* **Abstention accuracy:** unanswerable required facts the system correctly did not answer.
* **Question accuracy:** every required fact handled correctly and zero hallucinated claims.
* Mean LLM calls, tokens, estimated cost, and p50/p95 latency per question.

Intervals are 95% bootstrap CIs that resample **questions** (claims within a question are
correlated). Mock runs pool 5 seeds.

---

## Results

### Live model

**Not run yet.** No live numbers are reported here because none have been measured. To produce
them:

```bash
export ANTHROPIC_API_KEY=...            # never committed; read from the environment only
make eval-live LIMIT=50                 # cheap first pass
make eval-live                          # full held-out test split
```

This writes `results/anthropic_test.md` with the same table as below, plus a relative-change
line for the hallucinated-claim rate. The token counts in the offline run (≈930 tokens per
question for the baseline, ≈1,900 for multi-agent) come from the real prompts, so they give
a rough idea of what a live run will spend. The model and effort level are set with
`ANALYST_MODEL` and `ANALYST_EFFORT`.

### Offline pipeline check (simulated LLM)

> These numbers come from [`MockLLM`](src/analyst/llm/mock.py), a seeded behavioural
> simulator whose error rates are **chosen**, not learned (15% distractor misreads, 60%
> guess-instead-of-abstain, 50% injection compliance, 10% critic misses, 3% critic false
> rejects). They show that the critic loop, guardrails and judge behave as designed under
> those failure modes. **They are not evidence about any real model.**

Held-out `test` split, 200 questions × 5 seeds ([full report](results/mock_test.md)):

| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | Question accuracy ↑ | LLM calls / q | Est. cost / q |
|---|---|---|---|---|---|---|
| `single_agent` | 23.9% (20.4–27.7) | 86.3% | 27.6% | 68.3% | 1.00 | $0.0049 |
| `multi_agent` | 3.3% (2.0–4.8) | 93.0% | 88.9% | 89.7% | 3.89 | $0.0106 |
| `multi_agent_no_critic` | 23.1% (19.8–26.8) | 83.0% | 43.1% | 68.1% | 2.42 | $0.0065 |
| `multi_agent_no_guardrails` | 3.4% (2.1–4.9) | 92.9% | 88.0% | 89.4% | 3.93 | $0.0108 |

`adversarial` split, 18 questions × 5 seeds ([full report](results/mock_adversarial.md)):

| System | Hallucinated-claim rate ↓ | Question accuracy ↑ |
|---|---|---|
| `single_agent` | 20.3% | 73.3% |
| `multi_agent` | 1.7% | 93.3% |
| `multi_agent_no_critic` | 10.0% | 86.7% |
| `multi_agent_no_guardrails` | 15.7% | 70.0% |

What the ablations show about the *system* (these points hold whatever model is plugged in):

* **The critic loop does the work.** Taking out the critic puts the multi-agent system back
  at baseline hallucination levels. Decomposing the question alone doesn't help.
* **Guardrails matter only when the attack is retrieved.** On the general test split, removing
  quarantine changes almost nothing. On the adversarial split it's the difference between
  1.7% and 15.7%.
* **Reliability costs about 2x.** Multi-agent makes ~3.9 LLM calls per question vs 1, with ~2x
  the tokens and ~3.5x the latency. The trace for every run shows where that goes.

---

## Quickstart

```bash
git clone https://github.com/juhidixit/multi-agent-llm-analyst && cd multi-agent-llm-analyst
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev,api,live]"
make test             # 27 tests
make eval             # offline eval → results/mock_test.{md,json}
```

Ask one question and save its trace:

```bash
$ analyst ask "Which company had higher fiscal 2024 revenue, Jessamine Analytics or Ixora Dynamics, and by how much?" --trace-dir traces
Jessamine Analytics's fiscal 2024 revenue (USD millions) is 1,956.5. [galvan-labs-news] Ixora Dynamics's
fiscal 2024 revenue (USD millions) is 832.3. [calyx-energy-news] Jessamine Analytics is higher on fiscal
2024 revenue (USD millions) by 1,124.2 (computed).

6 LLM calls · 3076+368 tokens · est. $0.0197 · 8.54s
guardrails: quarantined jessamine-energy-forum: prompt-injection pattern

$ analyst trace traces/<id>.json
- pipeline.multi_agent (3.17 ms)
  - agent.planner (0.27 ms)
    - llm.planner (0.09 ms)  [219→74 tok, $0.002356]
  - agent.researcher (1.1 ms)
    - retrieve (0.36 ms)
    - llm.researcher (0.66 ms)  [594→61 tok, $0.003596]
  - agent.researcher (0.54 ms)
    - retrieve (0.28 ms)
    - llm.researcher (0.19 ms)  [566→55 tok, $0.003364]
  - agent.critic (0.19 ms)
    - llm.critic (0.13 ms)  [398→87 tok, $0.003332]
  - agent.researcher (0.61 ms)             ← revision after a critic rejection
    - retrieve (0.31 ms)
    - llm.researcher (0.23 ms)  [901→58 tok, $0.004764]
  - agent.critic (0.19 ms)
    - llm.critic (0.15 ms)  [398→33 tok, $0.002252]
totals: 6 LLM calls, 3076 in / 368 out tokens, $0.019664
```

(With the mock provider, span durations are real wall-clock time and the reported latency adds
a simulated 0.4 s + 60 tokens/s per call. Costs are estimates from token counts × the price
table in [`llm/pricing.py`](src/analyst/llm/pricing.py).)

### Using Claude

```bash
cp .env.example .env      # .env is git-ignored
export ANTHROPIC_API_KEY=... ANALYST_PROVIDER=anthropic
analyst ask "Where is Lumora Systems headquartered?"
```

The client ([`llm/anthropic_client.py`](src/analyst/llm/anthropic_client.py)) uses the official
SDK, reads credentials only from the environment, retries transient errors, and handles
refusals through server-side fallback.

### HTTP API

```bash
make serve   # or: docker build -t analyst . && docker run -p 8000:8000 analyst
curl -X POST localhost:8000/v1/analyze -H 'content-type: application/json' \
     -d '{"question": "For Lumora Systems, tell me who its CEO is and where it is headquartered.", "pipeline": "multi_agent"}'
```

Returns the answer, structured claims with citations, abstentions, guardrail events, and run
stats. Injection attempts get `422` and an exhausted budget gets `429`.

### Deploy to AWS

[`infra/terraform/main.tf`](infra/terraform/main.tf) provisions ECR, a Secrets Manager secret
for the API key, least-privilege IAM, and an App Runner service. See
[`docs/deploy-aws.md`](docs/deploy-aws.md).

---

## Repository layout

```
src/analyst/
  agents/        planner, researcher, critic (+ shared LLM-call / JSON-repair base)
  pipelines/     multi_agent, single_agent, synthesis
  corpus/        benchmark generator, BM25 retriever
  llm/           provider interface, Anthropic client, offline MockLLM, pricing
  eval/          judge, harness (bootstrap CIs), report
  memory.py      shared blackboard
  guardrails.py  injection / citation / PII / budget
  tracing.py     spans, token + cost accounting
  api.py, cli.py
data/            generated corpus, gold facts, dev / test / adversarial questions
results/         evaluation reports (JSON + Markdown)
tests/           unit + integration tests (pipelines, guardrails, judge, API, data integrity)
infra/terraform/ AWS App Runner deployment
```

## Limitations and next steps

* **No live-model numbers yet.** See [Live model](#live-model). The offline numbers are
  mechanics checks only.
* **Synthetic, narrow domain.** Seven attributes over fictional company facts. That keeps
  scoring exact, but real filings are messier. Next step: a held-out set over real 10-K
  excerpts with human-labelled claims.
* **Lexical retrieval.** BM25 is enough here. Hybrid dense + BM25 retrieval would matter on a
  real corpus.
* **Regex-based injection detection** catches known patterns, not novel ones. A classifier
  (or an LLM screen on retrieved text) would be the next layer.
* **The critic uses the same model as the researcher.** Using a different model, or self-
  consistency sampling, for the critic is a natural ablation.

## License

MIT
