"""Unit tests for hybrid retrieval: RRF fusion, BM25, rerank, and wiring."""

from __future__ import annotations

from rag.chunking import Chunk
from rag.retriever import (
    BM25Index,
    DenseIndex,
    HybridRetriever,
    ScoredChunk,
    lexical_rerank,
    reciprocal_rank_fusion,
)


def _fake_embed_factory():
    """Deterministic bag-of-words embedder over a fixed vocabulary (offline)."""
    vocab = ["mars", "moons", "phobos", "deimos", "wall", "photosynthesis", "oxygen"]

    def embed(texts):
        vectors = []
        for text in texts:
            low = text.lower()
            vectors.append([1.0 if term in low else 0.0 for term in vocab])
        return vectors

    return embed


# ------------------------------------------------------------------------ RRF
def test_rrf_rewards_items_ranked_high_in_both_lists():
    list_a = ["a", "b", "c"]
    list_b = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [cid for cid, _ in fused]
    # 'a' (1st,2nd) and 'b' (2nd,1st) appear in both -> must outrank c/d.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_rrf_top_rank_beats_single_high_rank():
    # 'x' is rank-1 in two lists; 'y' is rank-1 in one only.
    fused = reciprocal_rank_fusion([["x", "y"], ["x", "z"], ["y"]], k=60)
    assert fused[0][0] == "x"


def test_rrf_is_deterministic_on_ties():
    fused1 = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    fused2 = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    # Equal scores -> tie-broken by id, identical ordering regardless of input.
    assert [i for i, _ in fused1] == [i for i, _ in fused2] == ["a", "b"]


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert abs(fused["a"] - 1 / 61) < 1e-9
    assert abs(fused["b"] - 1 / 62) < 1e-9


# ----------------------------------------------------------------------- BM25
def test_bm25_ranks_keyword_match_first(sample_chunks):
    index = BM25Index(sample_chunks)
    hits = index.search("phobos deimos moons", top_k=3)
    top_id = hits[0][0]
    assert top_id == "mars.md::p1::c1"  # the Phobos/Deimos chunk


# --------------------------------------------------------------------- rerank
def test_lexical_rerank_promotes_query_term_overlap():
    chunks = [
        Chunk(text="completely unrelated text", doc_name="d", chunk_index=0),
        Chunk(text="mars red planet fourth", doc_name="d", chunk_index=1),
    ]
    scored = [ScoredChunk(chunks[0], 0.5), ScoredChunk(chunks[1], 0.4)]
    reranked = lexical_rerank("mars planet", scored)
    assert reranked[0].chunk.chunk_index == 1  # overlap beats raw score


# --------------------------------------------------------- end-to-end (sparse)
def test_hybrid_retriever_sparse_only_returns_relevant(sample_chunks):
    # No embed_fn -> sparse-only mode; still must surface the right chunk.
    retriever = HybridRetriever(sample_chunks)
    results = retriever.retrieve("how long is the great wall", top_k=2)
    assert results
    assert results[0].chunk.doc_name == "great_wall.md"


def test_dense_index_ranks_by_cosine_similarity(sample_chunks):
    # Pure-NumPy dense path with a deterministic fake embedder (no live server).
    index = DenseIndex(sample_chunks, _fake_embed_factory())
    hits = index.search("phobos and deimos moons", top_k=2)
    assert hits[0][0] == "mars.md::p1::c1"  # the Phobos/Deimos chunk
    assert hits[0][1] > 0.0  # cosine similarity is positive for a match


def test_dense_index_handles_empty_corpus():
    assert DenseIndex([], _fake_embed_factory()).search("anything", top_k=3) == []


def test_hybrid_retriever_dense_via_embed_fn(sample_chunks):
    # Building the real DenseIndex through embed_fn must work offline + fuse.
    retriever = HybridRetriever(sample_chunks, embed_fn=_fake_embed_factory())
    results = retriever.retrieve("mars moons phobos", top_k=3)
    ids = [r.chunk.chunk_id for r in results]
    assert "mars.md::p1::c1" in ids


def test_hybrid_retriever_fuses_dense_and_sparse(sample_chunks):
    # Inject a fake dense ranker to verify both signals are fused.
    class FakeDense:
        def search(self, query, top_k):
            return [("mars.md::p1::c0", 0.9), ("photosynthesis.md::p1::c0", 0.5)]

    retriever = HybridRetriever(sample_chunks, dense=FakeDense())
    results = retriever.retrieve("red planet", top_k=3)
    ids = [r.chunk.chunk_id for r in results]
    assert "mars.md::p1::c0" in ids
