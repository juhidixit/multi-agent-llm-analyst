# Evaluation: mock_adversarial

> **Offline run (simulated LLM).** These numbers come from `MockLLM`, a seeded behavioural simulator with *chosen* error rates (see `src/analyst/llm/mock.py`). They verify that the pipeline, critic loop, guardrails and judge behave as designed. They are **not** a measurement of any real model. Run `make eval-live` with an API key for real numbers.

Split `adversarial`, 18 questions × 5 seeds.

| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | Question accuracy ↑ | LLM calls / q | Tokens / q | Est. cost / q | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| `single_agent` | 20.3% (13.6–27.4) | 78.3% (69.2–86.1) | n/a | 73.3% (66.7–83.3) | 1.00 | 809 | $0.0044 | 1.35s / 2.22s |
| `multi_agent` | 1.7% (0.0–4.1) | 95.0% (91.3–98.3) | n/a | 93.3% (83.3–100.0) | 3.56 | 1531 | $0.0088 | 3.13s / 8.47s |
| `multi_agent_no_critic` | 10.0% (4.3–16.0) | 90.0% (84.0–95.7) | n/a | 86.7% (77.8–94.4) | 2.33 | 1026 | $0.0061 | 2.42s / 4.34s |
| `multi_agent_no_guardrails` | 15.7% (10.5–21.5) | 75.8% (68.8–81.8) | n/a | 70.0% (27.8–94.4) | 4.30 | 2144 | $0.0121 | 5.32s / 9.43s |

Brackets are 95% bootstrap intervals, resampling questions (claims from the same question are correlated, so questions are the unit). For question accuracy the bracket is the min–max across seeds.
