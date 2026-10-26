"""Small-sample statistics for success rates. Pure Python, no SciPy.

Planning papers report 50-episode success rates as point estimates. At n=50 a
95% interval is roughly +/-13 points, so most headline gaps are inside the
noise. Every number wm-eval prints carries an interval, and paired claims
(same episodes, two policies or two protocols) use an exact McNemar test.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def normalized(rate: float, floor: float, ceiling: float = 1.0) -> float | None:
    """Place ``rate`` between a floor (random policy) and a ceiling (oracle).

    0 means "no better than random on these episodes", 1 means "as good as the
    oracle". Undefined when the floor already reaches the ceiling.
    """
    span = ceiling - floor
    if span <= 1e-12:
        return None
    return (rate - floor) / span


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired outcomes (same episodes).

    Only discordant episodes carry information: ``b01`` where only ``b``
    succeeded and ``b10`` where only ``a`` succeeded.
    """
    if len(a) != len(b):
        raise ValueError("paired outcomes must have equal length")
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b01 = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = b10 + b01
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(b10, b01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"only_a": b10, "only_b": b01, "p_value": min(1.0, 2 * tail)}
