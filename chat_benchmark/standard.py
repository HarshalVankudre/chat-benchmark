"""One frozen, offline plain-text token unit for every provider.

This is a reference unit for useful-output throughput, never billing tokens.
The vocabulary and pre-tokenization pattern are frozen from OpenAI's MIT-licensed
o200k_base (see data/tiktoken-LICENSE). No model-name lookup or network is used.
"""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from math import isfinite
from pathlib import Path

from .workload import canonical_text, reference_digest

VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
PATTERN = "|".join([
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])
STANDARD_ID = "o200k_base-lf-v1:" + hashlib.sha256((VOCAB_SHA256 + PATTERN).encode()).hexdigest()


def metadata():
    return {"id": STANDARD_ID, "encoding": "o200k_base", "vocabulary_sha256": VOCAB_SHA256,
            "pattern_sha256": hashlib.sha256(PATTERN.encode()).hexdigest(),
            "normalization": "CRLF to LF; strip edge CR/LF; no added final newline",
            "scope": "Verified reference text only; full model-request seconds. Excludes reasoning and tool syntax from the numerator; includes their time. Not billing or engine decode tokens."}


@lru_cache(maxsize=1)
def tokenizer():
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError("Standard TPS requires tiktoken. Install requirements.txt with this Python interpreter before benchmarking.") from exc
    data = (Path(__file__).parent / "data" / "o200k_base.tiktoken").read_bytes()
    if hashlib.sha256(data).hexdigest() != VOCAB_SHA256:
        raise ValueError("Reference tokenizer vocabulary failed its SHA-256 check. Reinstall the benchmark.")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in data.splitlines())}
    return tiktoken.Encoding(name="chat-benchmark-o200k-base-v1", pat_str=PATTERN,
                            mergeable_ranks=ranks, special_tokens={})


@lru_cache(maxsize=64)
def reference_tokens(rows):
    # Counts reconstructed text only after a sample's original digest is checked.
    return len(tokenizer().encode_ordinary(canonical_text(rows)))


def standard_metrics(sample):
    result = {"standard_tokenizer": STANDARD_ID, "standard_tokens": None, "standard_tps": None}
    rows = sample.get("reference_rows")
    if (sample.get("status", "completed") != "completed" or sample.get("reference_matches") is not True
            or type(rows) is not int or not 1 <= rows <= 10000):
        return result
    continuity = sample.get("continuity")
    digest = (sample.get("payload_sha256") if continuity in ("single_file_payload", "unverified_file_payload") else
              sample.get("visible_sha256") if continuity == "single_text_block" else None)
    if digest != reference_digest(rows):
        return result
    count = reference_tokens(rows)
    duration = sample.get("duration_ms")
    result["standard_tokens"] = count
    if isinstance(duration, (int, float)) and isfinite(duration) and duration > 0:
        result["standard_tps"] = count * 1000 / duration
    return result
