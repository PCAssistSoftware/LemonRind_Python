"""Embedding vectors: storing them, and finding the ones closest in meaning.

An *embedding* is a list of numbers (here 1024 of them) that a special model computes for a piece of text,
so that texts with similar meaning get similar lists. "I live in Manchester" and "my home town is
Manchester" end up close together; "I like pizza" ends up further away. "Close" is measured with
**cosine similarity**: 1.0 means the same direction, around 0 means unrelated.

A trick makes this cheap: if every vector is scaled to length 1 (*normalised*), cosine similarity is just
the dot product. Comparing one query against thousands of stored vectors is then a single matrix
multiplication, which numpy does in microseconds. No vector database needed at this size.

Python ideas used here:

* ``numpy`` - fast arrays. ``matrix @ vector`` multiplies a whole table of vectors by one vector at once;
  there is no Python loop over the rows.
* ``bytes`` as a database value: a vector is stored in one SQLite ``BLOB`` column as its raw 4-byte floats
  (``float32``), 4 KB for 1024 numbers, rather than as text.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalise(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Scale a vector to length 1 (so a dot product between two of them is their cosine similarity)."""
    array = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(array))
    return array / length if length > 0 else array


def to_blob(vector: Sequence[float] | np.ndarray) -> bytes:
    """A normalised vector as raw bytes, for storing in a BLOB column."""
    return normalise(vector).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def rank(
    query: Sequence[float] | np.ndarray, vectors: Sequence[np.ndarray], *, limit: int | None = None
) -> list[tuple[int, float]]:
    """Compare ``query`` with every stored vector.

    Returns ``(position, similarity)`` pairs, most similar first. ``position`` is the index into
    ``vectors``, so the caller can look up whatever it stored alongside (the text, its id).
    """
    if not len(vectors):
        return []
    matrix = np.vstack(vectors)  # one row per stored vector
    similarities = matrix @ normalise(query)  # all the dot products at once
    order = np.argsort(-similarities)  # indexes, best first
    if limit is not None:
        order = order[:limit]
    return [(int(i), float(similarities[i])) for i in order]
