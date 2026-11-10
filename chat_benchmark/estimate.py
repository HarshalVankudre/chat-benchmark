"""Length-sweep estimators: separate steady generation speed from fixed overhead.

A request's wall time is modelled as  seconds = overhead + tokens / speed.
Overhead (queueing, time to first token, thinking, tool-call syntax, client and
helper start-up) is roughly constant across stages of one sweep, while the
generation term grows with the fixed reference length. Fitting a line through
several lengths therefore isolates the generation speed even when no per-token
timestamps are exposed, and it cancels constant client overhead in helper-clock
("bracket") timing. Nothing here invents data: too few points yield no estimate.
"""
from __future__ import annotations

from itertools import combinations
from math import isfinite
from random import Random
from statistics import mean, median

from .metrics import percentile

MIN_BOOTSTRAP_POINTS = 5


def _clean(points):
    return [(float(x), float(y)) for x, y in points
            if x is not None and y is not None and isfinite(x) and isfinite(y) and x > 0 and y > 0]


def pair_slopes(points):
    return [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in combinations(points, 2) if x1 != x2]


def theil_sen(points):
    """Median of pairwise slopes; one slow or fast sample cannot drag the line."""
    slopes = pair_slopes(points)
    if not slopes:
        return None
    slope = median(slopes)
    return slope, median(y - slope * x for x, y in points)


def _bootstrap(points, repetitions, seed=1729):
    rng, n, rates = Random(seed), len(points), []
    for _ in range(repetitions):
        sample = [points[rng.randrange(n)] for _ in range(n)]
        line = theil_sen(sample)
        if line and line[0] > 0:
            rates.append(1 / line[0])
    if len(rates) < repetitions * .8:
        return None
    return {"low": percentile(rates, .025), "high": percentile(rates, .975), "level": .95,
            "method": "percentile bootstrap of the Theil-Sen slope", "resamples": repetitions}


def fit_sweep(points, repetitions=2000):
    """Fit seconds = overhead + tokens / tps. Returns None without two distinct lengths."""
    points = _clean(points)
    line = theil_sen(points)
    lengths = sorted({x for x, _ in points})
    if line is None:
        return None
    slope, intercept = line
    result = {"method": "Theil-Sen line through (reference tokens, request seconds)",
              "points": len(points), "distinct_lengths": len(lengths),
              "token_range": [lengths[0], lengths[-1]],
              "seconds_per_token": slope, "overhead_seconds": intercept,
              "tps": None, "pairwise_tps": [], "pair_spread_pct": None, "r_squared": None,
              "max_residual_pct": None, "ci95": None, "status": "ok"}
    if slope <= 0:
        result["status"] = "non_positive_slope"
        return result
    slopes = pair_slopes(points)
    result["tps"] = 1 / slope
    result["pairwise_tps"] = sorted(1 / s for s in slopes if s > 0)
    if len(slopes) >= 2:
        # Every pair of lengths gives an independent speed. Their disagreement,
        # relative to the estimate, is the sweep's internal consistency check.
        result["pair_spread_pct"] = (max(slopes) - min(slopes)) / slope * 100
    residuals = [y - (intercept + slope * x) for x, y in points]
    result["max_residual_pct"] = max(abs(r) / y * 100 for r, (_, y) in zip(residuals, points))
    centre = mean(y for _, y in points)
    total = sum((y - centre) ** 2 for _, y in points)
    result["r_squared"] = 1 - sum(r * r for r in residuals) / total if total > 0 else None
    if len(points) >= MIN_BOOTSTRAP_POINTS and len(lengths) >= 3:
        result["ci95"] = _bootstrap(points, repetitions)
    return result


def agreement_pct(first, second):
    """Symmetric relative difference of two positive estimates, in percent."""
    if not first or not second or first <= 0 or second <= 0:
        return None
    return abs(first - second) / ((first + second) / 2) * 100


def grade(fit, basis, *, verified=True, cross_check_pct=None, flags=()):
    """Translate the checks into a plain confidence level with its reasons.

    high   - three or more lengths agree within 5% and nothing was flagged
    medium - lengths agree within 15%, or the checks are otherwise incomplete
    low    - two lengths only (no linearity check), or the lengths disagree
    """
    if not fit or fit.get("tps") is None:
        return {"level": "none", "reasons": ["fewer than two valid reference lengths, so generation speed cannot be separated from fixed overhead"]}
    reasons, level = [], "high"

    def cap(to, why):
        nonlocal level
        order = ("high", "medium", "low")
        if order.index(to) > order.index(level):
            level = to
        reasons.append(why)

    spread, lengths = fit.get("pair_spread_pct"), fit["distinct_lengths"]
    if lengths < 3 or spread is None:
        cap("low", "only two reference lengths: the estimate is exact for them but has no internal consistency check")
    elif spread > 15:
        cap("low", f"length pairs disagree by {spread:.1f}% (over 15%)")
    elif spread > 5:
        cap("medium", f"length pairs disagree by {spread:.1f}% (over 5%)")
    else:
        reasons.append(f"{lengths} reference lengths agree within {spread:.1f}%")
    if fit["overhead_seconds"] < -0.5:
        cap("medium", "fitted overhead is negative, so timing did not grow linearly with length")
    if basis == "bracket":
        reasons.append("helper-clock timing: constant client overhead cancels in the slope, variable delays (approval prompts, pauses) do not")
    elif basis != "native":
        cap("medium", "request starts come from session boundaries, which include client overhead")
    if cross_check_pct is not None:
        if cross_check_pct > 10:
            cap("medium", f"native and helper-clock estimates differ by {cross_check_pct:.1f}%")
        else:
            reasons.append(f"native and helper-clock estimates agree within {cross_check_pct:.1f}%")
    if not verified:
        cap("medium", "no transcript: literal generation and single tool call could not be verified")
    for flag in flags:
        cap("medium", flag)
    return {"level": level, "reasons": reasons}
