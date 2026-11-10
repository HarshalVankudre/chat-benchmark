"""Shared metric definitions. Missing data stays missing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, floor, isfinite
from statistics import mean, median, stdev
from random import Random


@dataclass
class Request:
    provider: str
    session_id: str
    request_id: str
    turn_id: str | None = None
    model: str | None = None
    effort: str | None = None
    service_tier: str | None = None
    speed: str | None = None
    requested_model: str | None = None
    started_ms: float | None = None
    completed_ms: float | None = None
    duration_ms: float | None = None
    ttft_ms: float | None = None
    first_content_ms: float | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    attempts: int | None = None
    status: str = "completed"
    timing_source: str = "not exposed"
    ttft_source: str = "not exposed"
    settings_source: str = "session"
    run_id: str | None = None
    sample_index: int | None = None
    payload_sha256: str | None = None
    notes: list[str] = field(default_factory=list)
    context_epoch: int = 0
    phase: str = "measured"
    client_version: str | None = None
    visible_words: int | None = None
    text_span_ms: float | None = None
    text_span_source: str = "not exposed"
    text_block_count: int | None = None
    continuity: str = "unknown"
    target_seconds: float | None = None
    visible_sha256: str | None = None
    reference_matches: bool | None = None
    reference_rows: int | None = None
    client_estimated_usd: float | None = None
    stop_reason: str | None = None
    tool_call_count: int | None = None
    payload_words: int | None = None
    payload_bytes: int | None = None
    temporary_cleanup: str | None = None
    # Helper-clock interval: previous helper exit -> this helper's process start.
    # Available for every client; includes client and tool-dispatch overhead.
    bracket_ms: float | None = None
    # Native duration of the tool-argument stream, when the client records it.
    stream_ms: float | None = None
    stream_source: str = "not exposed"
    # Reference rows found literally in the transcript's tool-call arguments.
    tool_input_rows: int | None = None
    provenance: str = "unverified"
    usage_source: str = "client transcript"

    @property
    def timing_basis(self) -> str:
        if self.timing_source.startswith("native "):
            return "native"
        if self.timing_source.startswith("helper bracket"):
            return "bracket"
        return "boundary"

    @property
    def prestream_ms(self) -> float | None:
        """Latency before the payload started streaming (TTFT plus any thinking)."""
        if self.stream_ms is None or self.duration_ms is None or self.timing_basis != "native":
            return None
        value = self.duration_ms - self.stream_ms
        return value if value >= 0 else None

    @property
    def effective_tps(self) -> float | None:
        """All reported output tokens / full model request latency."""
        if self.output_tokens is None or self.output_tokens < 0 or not self.duration_ms or self.duration_ms <= 0 or not isfinite(self.duration_ms):
            return None
        return self.output_tokens * 1000 / self.duration_ms

    @property
    def post_ttft_tps(self) -> float | None:
        """A derived rate, available only with zero reported reasoning tokens.

        Never put hidden reasoning in the numerator after a text-first boundary.
        The final completion event can lag the final token, so this is explicitly
        a client-side derived rate, not an engine decoding-speed measurement.
        """
        if self.reasoning_tokens != 0 or self.output_tokens is None:
            return None
        if self.ttft_ms is None or self.duration_ms is None:
            return None
        window = self.duration_ms - self.ttft_ms
        if window <= 0 or self.output_tokens <= 1:
            return None
        return (self.output_tokens - 1) * 1000 / window

    @property
    def cache_hit_pct(self) -> float | None:
        if self.input_tokens is None or self.cached_input_tokens is None or self.input_tokens <= 0:
            return None
        return self.cached_input_tokens * 100 / self.input_tokens

    @property
    def visible_wps(self):
        return self.visible_words * 1000 / self.text_span_ms if self.visible_words and self.text_span_ms and self.text_span_ms > 0 else None

    @property
    def reference_rows_per_second(self):
        if self.status != "completed" or self.reference_matches is not True or not self.reference_rows or not self.duration_ms or self.duration_ms <= 0:
            return None
        return self.reference_rows * 1000 / self.duration_ms

    @property
    def payload_bytes_per_second(self):
        if self.reference_rows_per_second is None or self.payload_bytes is None:
            return None
        return self.payload_bytes * 1000 / self.duration_ms

    def settings(self) -> dict:
        return {key: getattr(self, key) for key in ("model", "effort", "service_tier", "speed")}

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(effective_tps=self.effective_tps, post_ttft_tps=self.post_ttft_tps,
                      cache_hit_pct=self.cache_hit_pct, visible_wps=self.visible_wps,
                      reference_rows_per_second=self.reference_rows_per_second,
                      payload_bytes_per_second=self.payload_bytes_per_second,
                      timing_basis=self.timing_basis, prestream_ms=self.prestream_ms)
        return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = floor(position), ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def block_interval(values: list[float], repetitions=1024) -> dict | None:
    """Circular moving-block bootstrap median interval with a fixed seed.

    Adjacent samples are resampled together; this still assumes approximate local
    stationarity. We expose drift separately rather than imply a formal guarantee.
    """
    n = len(values)
    if n < 10:
        return None
    rng, length, estimates = Random(1729), max(2, ceil(n ** .5)), []
    for _ in range(repetitions):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[(start + j) % n] for j in range(length))
        estimates.append(median(sample[:n]))
    return {"low": percentile(estimates, .025), "high": percentile(estimates, .975),
            "method": "circular moving-block bootstrap", "level": .95,
            "block_length": length, "resamples": repetitions}


def stats(values, intervals=True) -> dict:
    values = [float(v) for v in values if v is not None and isfinite(v)]
    center = median(values) if values else None
    interval = block_interval(values) if intervals else None
    deviation = stdev(values) if len(values) > 1 else None
    average = mean(values) if values else None
    mad = median(abs(v - center) for v in values) if values else None
    return {"n": len(values), "mean": average, "median": center,
            "p95": percentile(values, .95), "p99": percentile(values, .99) if len(values) >= 100 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "stdev": deviation, "cv_pct": deviation / average * 100 if deviation is not None and average else None,
            "mad": mad, "outliers_retained": (sum(abs(v - center) > 3.5 * 1.4826 * mad for v in values)
                if mad else sum(v != center for v in values)),
            "median_ci95": interval,
            "median_ci_relative_half_width_pct": ((interval["high"] - interval["low"]) * 50 / abs(center)
                                                  if interval and center else None)}


def summarize(requests: list[Request], expected_settings: dict | None = None, intervals=True) -> dict:
    """Keep different model/effort/tier configurations out of the same aggregate."""
    drift, valid = [], []
    expected_settings = dict(expected_settings or {})
    # A fresh chat can expose settings only once its first response is persisted.
    # Fill unknowns from this run's actual samples, never from global defaults.
    for request in requests:
        for key, value in request.settings().items():
            if expected_settings.get(key) is None and value is not None:
                expected_settings[key] = value
    for request in requests:
        changed = [k for k, v in expected_settings.items()
                   if v is not None and getattr(request, k, None) is not None
                   and getattr(request, k) != v]
        if changed:
            drift.append({"request_id": request.request_id, "fields": changed,
                          "settings": request.settings()})
        elif request.status == "completed":
            valid.append(request)
    timed = [r for r in valid if r.duration_ms is not None and r.duration_ms > 0
             and r.output_tokens is not None]
    total_duration = sum(r.duration_ms for r in timed)
    return {
        "settings": expected_settings,
        "requests": len(requests), "valid_requests": len(valid),
        "failed_requests": sum(r.status == "failed" for r in requests),
        "settings_drift": drift,
        "effective_tps": stats((r.effective_tps for r in valid), intervals),
        "post_ttft_tps": stats((r.post_ttft_tps for r in valid), intervals),
        "ttft_ms": stats((r.ttft_ms for r in valid), intervals),
        "duration_ms": stats((r.duration_ms for r in valid), intervals),
        "weighted_effective_tps": (sum(r.output_tokens for r in timed) * 1000 / total_duration
                                   if total_duration > 0 else None),
        "output_tokens": (sum(r.output_tokens for r in valid)
                          if valid and all(r.output_tokens is not None for r in valid) else None),
        "input_tokens": (sum(r.input_tokens for r in valid)
                         if valid and all(r.input_tokens is not None for r in valid) else None),
        "reasoning_tokens": (sum(r.reasoning_tokens for r in valid)
                             if valid and all(r.reasoning_tokens is not None for r in valid) else None),
    }
