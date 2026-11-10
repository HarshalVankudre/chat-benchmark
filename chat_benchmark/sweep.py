"""Turn a sweep's samples into the headline numbers. Works on saved report rows,
so earlier verified runs gain the same analysis without new model output."""
from __future__ import annotations

from statistics import median

from .estimate import agreement_pct, fit_sweep, grade

DEFINITION = ("Generation speed = reference tokens per second once output is flowing: the slope of request "
              "time against fixed-text length across the sweep's stages. Fixed per-request overhead (queueing, "
              "time to first token, thinking, tool-call syntax, client latency) falls into the intercept instead "
              "of diluting the rate.")
VERIFIED_STRUCTURES = ("single_file_payload", "single_text_block")


def basis_of(sample) -> str:
    if sample.get("timing_basis"):
        return sample["timing_basis"]
    source = sample.get("timing_source") or ""
    return "native" if source.startswith("native ") else "bracket" if source.startswith("helper bracket") else "boundary"


def valid_samples(samples, excluded=()):
    return [s for s in samples if s.get("phase") != "warmup" and s.get("request_id") not in excluded
            and s.get("status", "completed") == "completed" and s.get("standard_tokens") is not None]


def _points(valid, value):
    """(tokens, seconds) for every valid sample, or None unless all of them have the value."""
    pairs = [(s["standard_tokens"], value(s)) for s in valid]
    if not pairs or any(not isinstance(y, (int, float)) or y <= 0 for _, y in pairs):
        return None
    return [(x, y / 1000) for x, y in pairs]


def first_token(sample):
    """(milliseconds, kind) of the latency before output: native TTFT, else derived."""
    if sample.get("ttft_ms") is not None:
        return sample["ttft_ms"], "native"
    if sample.get("prestream_ms") is not None:
        return sample["prestream_ms"], "derived"
    return None, None


def generation_summary(samples, excluded=()):
    valid = valid_samples(samples, excluded)
    candidates = {
        "native": _points(valid, lambda s: s.get("duration_ms") if basis_of(s) == "native" else None),
        "bracket": _points(valid, lambda s: s.get("bracket_ms") if s.get("bracket_ms") is not None
                           else (s.get("duration_ms") if basis_of(s) == "bracket" else None)),
        "stream": _points(valid, lambda s: s.get("stream_ms")),
        "boundary": _points(valid, lambda s: s.get("duration_ms")),
    }
    basis = next((name for name in ("native", "bracket", "boundary") if candidates[name]), None)
    fits = {name: fit_sweep(points) for name, points in candidates.items() if points}
    fit = fits.get(basis) if basis else None
    result = {"definition": DEFINITION, "basis": basis, "valid_samples": len(valid),
              "tps": None, "overhead_seconds": None, "seconds_per_token": None, "pair_spread_pct": None,
              "pairwise_tps": [], "ci95": None, "points": len(valid), "distinct_lengths": 0,
              "r_squared": None, "max_residual_pct": None, "native_token_tps": None,
              "single_stage_tps": None, "cross_check": None, "helper_clock_overhead_ms": None}
    if fit:
        result.update({key: fit[key] for key in ("tps", "overhead_seconds", "seconds_per_token", "pair_spread_pct",
                                                 "pairwise_tps", "ci95", "points", "distinct_lengths",
                                                 "r_squared", "max_residual_pct")})
        result["fit_status"] = fit["status"]
    # The same slope in each provider's own token unit, for audit against native counts.
    seconds = dict(zip((id(s) for s in valid), (y for _, y in candidates[basis]))) if basis else {}
    native_points = [(s["output_tokens"] - (s.get("reasoning_tokens") or 0), seconds[id(s)]) for s in valid
                     if isinstance(s.get("output_tokens"), (int, float))]
    if basis and len(native_points) == len(valid):
        native_fit = fit_sweep(native_points)
        result["native_token_tps"] = native_fit["tps"] if native_fit else None
    native_tps = (fits.get("native") or {}).get("tps")
    helper_tps = (fits.get("bracket") or {}).get("tps")
    stream_tps = (fits.get("stream") or {}).get("tps")
    cross = agreement_pct(native_tps, helper_tps)
    if native_tps or helper_tps or stream_tps:
        result["cross_check"] = {"native_tps": native_tps, "helper_clock_tps": helper_tps,
                                 "native_stream_tps": stream_tps, "native_vs_helper_clock_pct": cross,
                                 "native_vs_stream_pct": agreement_pct(native_tps, stream_tps)}
    overheads = [s["bracket_ms"] - s["duration_ms"] for s in valid
                 if basis_of(s) == "native" and s.get("bracket_ms") is not None and s.get("duration_ms")]
    if overheads:
        result["helper_clock_overhead_ms"] = {"median": median(overheads), "min": min(overheads), "max": max(overheads)}
    if result["tps"] is None and len(valid) >= 1:
        # One length cannot separate overhead; fall back to the rate after the first token.
        sample = max(valid, key=lambda s: s["standard_tokens"])
        before, kind = first_token(sample)
        duration = sample.get("duration_ms")
        if before is not None and duration and duration > before and basis_of(sample) == "native":
            result["single_stage_tps"] = {"tps": sample["standard_tokens"] * 1000 / (duration - before),
                                          "tokens": sample["standard_tokens"], "first_token": kind,
                                          "note": "single stage after the first token; still includes tool-call syntax and any thinking"}
    flags = []
    if any((s.get("visible_words") or 0) > 40 for s in valid if basis_of(s) != "bracket"):
        flags.append("some responses carried extra visible text, which makes the overhead less constant")
    thinking = [s.get("reasoning_tokens") or 0 for s in valid]
    if valid and max(thinking) - min(thinking) > .1 * min(s["standard_tokens"] for s in valid):
        flags.append(f"thinking varied between stages ({min(thinking):,.0f}-{max(thinking):,.0f} reasoning tokens), so overhead was not constant")
    if overheads and max(overheads) - min(overheads) > 3000 and basis == "bracket":
        flags.append("client delay between helper commands varied by more than 3 s")
    verified = bool(valid) and all(s.get("continuity") in VERIFIED_STRUCTURES for s in valid)
    result["confidence"] = grade(fit, basis, verified=verified, cross_check_pct=cross, flags=flags)
    return result


def end_to_end(samples, excluded=()):
    """The longest verified stage: the least overhead-diluted single-request rate."""
    valid = [s for s in valid_samples(samples, excluded) if s.get("standard_tps") is not None]
    if not valid:
        return None
    sample = max(valid, key=lambda s: s["standard_tokens"])
    return {"tokens": sample["standard_tokens"], "rows": sample.get("reference_rows"),
            "tps": sample["standard_tps"], "seconds": sample["duration_ms"] / 1000}


def first_token_summary(samples, excluded=()):
    values = [first_token(s) for s in valid_samples(samples, excluded)]
    values = [(v, kind) for v, kind in values if v is not None]
    if not values:
        return None
    kinds = {kind for _, kind in values}
    return {"median_ms": median(v for v, _ in values), "n": len(values),
            "source": "native" if kinds == {"native"} else "derived" if kinds == {"derived"} else "mixed"}
