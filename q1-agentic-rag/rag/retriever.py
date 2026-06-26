"""Hybrid retrieval: dense (NumPy cosine) + sparse (BM25) fused with RRF, rerank.

Pipeline
--------
1. **Sparse** -- BM25 over chunk tokens (great for rare keywords / exact terms).
2. **Dense**  -- cosine similarity over Ollama embeddings stored in a NumPy
   matrix (great for paraphrase / semantic matches). No external vector DB --
   chunks are embedded once at build time, then queries are a single matmul.
3. **Fusion** -- Reciprocal Rank Fusion combines the two ranked lists into one
   robust ordering without needing comparable score scales.
4. **Rerank** -- a lightweight lexical-overlap reranker breaks ties and pushes
   chunks that share query terms upward before the agent grades them.

The dense index needs only NumPy plus an embedding function -- no heavyweight or
build-fragile vector store. Both rankers are injectable, so the fusion path is
trivially testable with fakes and unit tests never need a live embedder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

EmbedFn = Callable[[Sequence[str]], list[list[float]]]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    """A chunk paired with its fused retrieval score (higher = better)."""

    chunk: Chunk
    score: float


class Ranker(Protocol):
    """Anything that can rank chunk ids for a query."""

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ...


# --------------------------------------------------------------------- fusion
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one using Reciprocal Rank Fusion.

    For each list, an item at 0-based ``rank`` contributes ``1 / (k + rank + 1)``
    to its id's score. Scores are summed across lists and sorted descending.
    Ties break deterministically by id so output is stable for tests.

    Args:
        rankings: ordered id lists, each best-first.
        k: RRF dampening constant; larger ``k`` flattens rank influence.

    Returns:
        ``[(id, fused_score), ...]`` sorted best-first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ rerankers
def lexical_rerank(
    query: str, scored: Sequence[ScoredChunk]
) -> list[ScoredChunk]:
    """Re-order ``scored`` by blending fused score with query-term overlap.

    A cheap, dependency-free reranker: chunks containing more distinct query
    terms float up. Keeps everything deterministic and offline-friendly.
    """
    q_terms = set(_tokenize(query))
    if not q_terms:
        return list(scored)

    def boost(sc: ScoredChunk) -> float:
        terms = set(_tokenize(sc.chunk.text))
        overlap = len(q_terms & terms) / len(q_terms)
        return sc.score + overlap

    return sorted(scored, key=lambda sc: -boost(sc))


# ------------------------------------------------------------------ sub-index
class BM25Index:
    """Sparse keyword ranker backed by ``rank_bm25``."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._ids = [c.chunk_id for c in self._chunks]
        corpus = [_tokenize(c.text) for c in self._chunks] or [[""]]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._ids, scores), key=lambda kv: -kv[1]
        )
        return [(cid, float(s)) for cid, s in ranked[:top_k]]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise ``matrix`` so dot products equal cosine similarities."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for empty vectors
    return matrix / norms


class DenseIndex:
    """Dense vector ranker over a NumPy embedding matrix (cosine similarity).

    All chunks are embedded once at construction; each query embeds a single
    string and scores it against every chunk via one normalised matmul. This is
    exact (no ANN approximation) and plenty fast for the document scales here,
    while avoiding any external vector-database dependency.
    """

    def __init__(self, chunks: Sequence[Chunk], embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        self._ids = [c.chunk_id for c in chunks]
        if chunks:
            embeddings = embed_fn([c.text for c in chunks])
            self._matrix = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        query_vec = np.asarray(self._embed_fn([query])[0], dtype=np.float32)
        query_vec = _l2_normalize(query_vec.reshape(1, -1))[0]
        # Cosine similarity = dot product of L2-normalised vectors.
        sims = self._matrix @ query_vec
        k = min(top_k, len(self._ids))
        # argpartition for the top-k, then sort just those descending.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._ids[i], float(sims[i])) for i in top_idx]


# ----------------------------------------------------------------- retriever
class HybridRetriever:
    """Combine a sparse and a dense ranker with RRF + a rerank pass."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        sparse: Ranker | None = None,
        dense: Ranker | None = None,
        embed_fn: EmbedFn | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, Sequence[ScoredChunk]], list[ScoredChunk]]
        | None = lexical_rerank,
    ) -> None:
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self._chunks}
        self._sparse = sparse or BM25Index(self._chunks)
        if dense is not None:
            self._dense: Ranker | None = dense
        elif embed_fn is not None:
            self._dense = DenseIndex(self._chunks, embed_fn)
        else:
            # No embedder provided -> sparse-only mode (still fully functional).
            self._dense = None
        self._rrf_k = rrf_k
        self._reranker = reranker

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top ``top_k`` chunks for ``query`` after fusion + rerank.

        Each sub-ranker is queried for a wider candidate pool (``top_k * 3``) so
        RRF has enough overlap to work with before truncating to ``top_k``.
        """
        pool = max(top_k * 3, top_k)
        rankings: list[list[str]] = []

        sparse_hits = self._sparse.search(query, pool)
        rankings.append([cid for cid, _ in sparse_hits])
        if self._dense is not None:
            dense_hits = self._dense.search(query, pool)
            rankings.append([cid for cid, _ in dense_hits])

        fused = reciprocal_rank_fusion(rankings, k=self._rrf_k)
        scored = [
            ScoredChunk(chunk=self._by_id[cid], score=score)
            for cid, score in fused
            if cid in self._by_id
        ]
        if self._reranker is not None:
            scored = self._reranker(query, scored)
        return scored[:top_k]
