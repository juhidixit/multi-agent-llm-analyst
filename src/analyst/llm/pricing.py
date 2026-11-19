"""USD per million tokens (input, output). Update when provider pricing changes."""

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    # Possible server-side fallback targets (see anthropic_client.py).
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def price(model: str) -> tuple[float, float]:
    """Exact match, else the longest known ID the model name starts with (dated snapshots)."""
    if model in PRICES:
        return PRICES[model]
    keys = [k for k in PRICES if model.startswith(k)]
    return PRICES[max(keys, key=len)] if keys else (0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price(model)
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
