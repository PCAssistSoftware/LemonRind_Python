"""Splitting a long text into overlapping chunks that can each be embedded and searched.

Why chunk at all? An embedding is one vector for a whole piece of text. For a 50-page document, one vector
cannot say "page 31 is about refunds"; it can only say something vague about the whole. Split the document
into pieces of a few hundred words and each piece gets its own vector, so a question finds the *piece* that
answers it, and only that piece is given to the model (not the whole document, which might not even fit).

**Overlap** matters because a cut can land in the middle of the answer: if chunk 1 ends mid-explanation and
chunk 2 begins after it, neither contains the whole thought. Repeating the last words of each chunk at the
start of the next means a sentence near a boundary appears whole in at least one of them.

This is *word-count* chunking: simple and predictable. (A smarter splitter would respect headings and
paragraphs; the overlap covers most of what that would buy.)

Python ideas used here:

* ``re.finditer`` returns *match objects* with positions, so the chunk can be cut out of the original text
  with a slice. Splitting on whitespace and re-joining would flatten the line breaks the text had.
* ``range(start, stop, step)`` to walk through the words in strides.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def chunk_text(text: str, words_per_chunk: int = 400, overlap: int = 60) -> list[str]:
    """Split ``text`` into chunks of about ``words_per_chunk`` words, each starting ``overlap`` words
    before the previous one ended. Text shorter than one chunk comes back as a single chunk; empty text
    gives an empty list. Line breaks inside a chunk are preserved."""
    if words_per_chunk < 1:
        raise ValueError("words_per_chunk must be at least 1")
    if not 0 <= overlap < words_per_chunk:
        raise ValueError("overlap must be at least 0 and smaller than words_per_chunk")

    words = list(
        _WORD.finditer(text)
    )  # each match knows where its word starts and ends in ``text``
    if not words:
        return []

    chunks: list[str] = []
    stride = words_per_chunk - overlap
    for start in range(0, len(words), stride):
        last = min(start + words_per_chunk, len(words)) - 1
        chunks.append(text[words[start].start() : words[last].end()])
        if last == len(words) - 1:  # this chunk reached the end: a further one would only repeat it
            break
    return chunks
