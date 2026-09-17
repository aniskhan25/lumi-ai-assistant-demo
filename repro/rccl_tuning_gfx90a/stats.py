#!/usr/bin/env python3
"""The statistics the study needs, in the standard library only.

No numpy or scipy: this has to give identical answers inside the container and on
a laptop, because the laptop is where the analysis is checked against synthetic
data with a known injected effect before any real data exists.

Everything works on **log** metrics. Collective times are positive and roughly
multiplicative in their noise, so a difference of logs is a ratio, which is what a
"4% regression" actually means.
"""
from __future__ import annotations

import math
from statistics import fmean, median, stdev

# Two-sided 95% Student-t, indexed by degrees of freedom. Beyond 30 the normal
# approximation is closer than anything else in this pipeline is accurate to.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042}


def t95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.960
    return _T95[min(k for k in _T95 if k >= df)]


def geomean(values: list[float]) -> float | None:
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return math.exp(fmean(math.log(v) for v in positive))


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between order statistics; matches numpy's default."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = q * (len(data) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def cv_percent(log_values: list[float]) -> float | None:
    """Coefficient of variation from log-scale spread: exp(sd) - 1, as a percent."""
    if len(log_values) < 2:
        return None
    return (math.exp(stdev(log_values)) - 1.0) * 100.0


def mean_ci(values: list[float]) -> tuple[float, float] | None:
    """(mean, half-width) of a two-sided 95% interval. Values are log ratios."""
    if not values:
        return None
    if len(values) == 1:
        return values[0], float("inf")
    return fmean(values), t95(len(values) - 1) * stdev(values) / math.sqrt(len(values))


def as_percent(log_ratio: float) -> float:
    """A log ratio as the percent change it represents."""
    return (math.exp(log_ratio) - 1.0) * 100.0


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """(slope, intercept) by least squares. Used for the position covariate."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = fmean(xs), fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope, my - slope * mx


def variance_components(blocks: list[list[float]]) -> dict:
    """Split log-scale variance into allocation, position and residual parts.

    `blocks` is one list of log metrics per allocation, in slot order. A one-way
    nested decomposition: between-block spread is sigma_alloc, within-block spread
    is sigma_pos. sigma_within comes from the raw samples and is passed in
    separately, because it lives a level below this.
    """
    usable = [b for b in blocks if len(b) >= 2]
    if len(usable) < 2:
        return {"error": "need at least 2 allocations with 2+ slots each"}

    block_means = [fmean(b) for b in usable]
    within = [v - m for b, m in zip(usable, block_means) for v in b]
    return {
        "n_alloc": len(usable),
        "n_slots": sum(len(b) for b in usable),
        "cv_alloc_pct": cv_percent(block_means),
        "cv_pos_pct": cv_percent(within) if len(within) > 1 else None,
    }


def mde_percent(cv_pos_pct: float, k: int) -> float:
    """Smallest paired effect detectable at 80% power, alpha=0.05, k allocations.

    Pairing against an in-block sentinel cancels the allocation term, so the
    residual on a paired log ratio is sqrt(2)*sigma_pos. 2.8 is z(0.975)+z(0.80).
    """
    return 2.8 * math.sqrt(2.0) * cv_pos_pct / math.sqrt(max(k, 1))


def lenth_pse(effects: list[float]) -> float | None:
    """Lenth's pseudo standard error: a spread estimate that survives a few real
    effects hiding among many null ones, which is exactly the screening situation."""
    if len(effects) < 3:
        return None
    s0 = 1.5 * median(abs(e) for e in effects)
    kept = [abs(e) for e in effects if abs(e) < 2.5 * s0]
    if not kept:
        return s0
    return 1.5 * median(kept)


def benjamini_hochberg(pvalues: list[float], q: float = 0.10) -> list[bool]:
    """Which hypotheses survive at false-discovery rate q, order preserved."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= q * rank / n:
            cutoff = rank
    keep = [False] * n
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff:
            keep[idx] = True
    return keep


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal, via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))
