# Requirement → Implementation Mapping

Every assignment requirement (and bonus) mapped to the exact file/function that
satisfies it. All paths are relative to `q1-agentic-rag/`.

## Core requirements

| # | Requirement | Where it lives |
|---|-------------|----------------|
| 1 | **Agentic RAG** with a reasoning loop that decides actions | `rag/agent.py` → `AgenticRAG.run` (hand-rolled state machine) |
| 1a | Query analysis / rewriting | `rag/agent.py` → `AgenticRAG._rewrite_query` (step `analyze_query`) |
| 1b | Retrieval tool call | `rag/agent.py` → `run` calls `HybridRetriever.retrieve` (step `retrieve`) |
| 1c | Self-grading / relevance check on retrieved chunks | `rag/agent.py` → `AgenticRAG.grade_chunks` / `_grade_chunk` (step `grade`) |
| 1d | Query re-formulation + re-retrieval when chunks are weak | `rag/agent.py` → `_reformulate_query`, loop in `run` (step `reformulate`) |
| 1e | Final grounded answer | `rag/agent.py` → `AgenticRAG._generate_answer` (step `generate`) |
| 1f | Retrieves correct chunks | `rag/retriever.py` → `HybridRetriever.retrieve`; verified by `eval/eval.py` (100% retrieval hit-rate on samples) |
| 2 | Working **Streamlit** prototype: upload/sample docs, ask, see answer + reasoning + chunks | `app.py` |
| 5 | Test suite | `tests/` (chunking, retrieval fusion, citations, agent logic) + `eval/` |

## Bonus requirements

| # | Bonus | Where it lives |
|---|-------|----------------|
| 3 | **Citations** as `[doc_name p.X / chunk N]` | `rag/citations.py` → `format_citation`, enforced by `_ANSWER_SYS` prompt in `rag/agent.py` |
| 3a | UI shows exact source passages backing the answer | `app.py` → "Source passages" expander; `rag/citations.py` → `used_passages` |
| 4 | **Hybrid retrieval** = dense (NumPy cosine) + sparse (BM25) | `rag/retriever.py` → `DenseIndex`, `BM25Index` |
| 4a | **Reciprocal Rank Fusion** | `rag/retriever.py` → `reciprocal_rank_fusion` |
| 4b | Re-ranking / relevance grading step | `rag/retriever.py` → `lexical_rerank`; plus agent self-grading in `rag/agent.py` |
| 4c | Sensible chunking (recursive/sentence-aware + overlap) | `rag/chunking.py` → `chunk_text`, `_split_recursive`, `_merge_with_overlap` |

## Supporting deliverables

| Item | Where it lives |
|------|----------------|
| Ollama client wrapper (chat + embeddings) | `rag/llm.py` → `LLMClient` |
| Config via `.env` (`python-dotenv`) | `rag/config.py` → `Config`, `get_config`; `.env.example` |
| Document ingestion (txt/md/pdf, per-page) | `rag/ingest.py` |
| Pipeline wiring (docs → retriever → agent) | `rag/pipeline.py` → `build_agent`, `build_agent_from_dir` |
| Pinned dependencies | `requirements.txt` |
| Bundled sample docs (known ground truth) | `data/photosynthesis.md`, `data/mars.md`, `data/great_wall.md` |

## Test-case mapping (requirement 5)

| Test target | File |
|-------------|------|
| Chunking (size, overlap, indices, ids) | `tests/test_chunking.py` |
| Retrieval fusion — **RRF ordering & score formula** | `tests/test_retriever.py` |
| BM25 ranking + lexical rerank + hybrid fusion | `tests/test_retriever.py` |
| Citation formatting / extraction / filtering | `tests/test_citations.py` |
| Agent grade + decision + reformulation logic (mocked LLM) | `tests/test_agent.py` |
| Live end-to-end (auto-skip if no Ollama) | `tests/test_integration.py` (`@pytest.mark.integration`) |
| Eval harness: retrieval hit-rate + groundedness | `eval/eval.py` + `eval/qa.jsonl` |

**Test status:** `python -m pytest -q` → 27 passed, 1 deselected (integration).
**Eval status:** `python -m eval.eval` (mock) → retrieval hit-rate 100%, groundedness 83%.
