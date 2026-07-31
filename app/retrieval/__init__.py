"""Local corpus retrieval (ROADMAP step 4).

The last error class the other steps cannot touch: things neither model knows, where
extra thinking does not help because the fact was never in the weights. It is also the
one place a small model genuinely beats a large one — reading comprehension is something
a 1.2B is good at, recall is not, so a grounded 1B can beat an ungrounded 14B on a
factual question.

Deliberately gated rather than always-on. In the paper behind this plan, indiscriminate
retrieval *hurt* GPQA by 5.0 points, because retrieved passages overrode correct
parametric knowledge. Two gates therefore stand in front of it: a cheap syntactic one in
app/effort.py that decides whether the prompt is even a lookup, and a similarity
threshold here that decides whether what came back is actually about the question.
"""

from app.retrieval.chunk import Chunk, chunk_markdown
from app.retrieval.service import Retrieval, RetrievalResult
from app.retrieval.store import Hit, VectorStore

__all__ = [
    "Chunk",
    "Hit",
    "Retrieval",
    "RetrievalResult",
    "VectorStore",
    "chunk_markdown",
]
