"""A public fixed reference; generation is performed by the chat, not this module."""
import hashlib

WORDS = "amber birch cedar delta ember frost grove harbor"


def rows_for(seconds):
    return max(1, round(seconds * 6.4))


def canonical_text(rows):
    return "\n".join(f"{i:04d} {WORDS}" for i in range(1, rows + 1))


def text_digest(text):
    # Normalize line endings and a final newline; content and internal whitespace
    # must otherwise match exactly. No chat text is persisted.
    return hashlib.sha256(text.replace("\r\n", "\n").strip("\r\n").encode("utf-8")).hexdigest()


def reference_digest(rows):
    return text_digest(canonical_text(rows))
