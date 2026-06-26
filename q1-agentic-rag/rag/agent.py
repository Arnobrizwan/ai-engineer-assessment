"""The agentic RAG reasoning loop.

This is a hand-rolled, fully-observable state machine (chosen over a heavier
graph framework for reliability and easy testing). Each turn the agent decides
its next action rather than running a fixed pipeline:

    analyze -> retrieve -> grade -> [reformulate -> retrieve -> grade]* -> answer

States
------
* **analyze_query**   -- rewrite the user question into a retrieval-optimised query.
* **retrieve**        -- hybrid (dense+sparse+RRF+rerank) retrieval.
* **grade**           -- self-grade each chunk's relevance (LLM yes/no).
* **decide**          -- enough good evidence? if not and budget remains,
                         reformulate the query and loop; otherwise answer.
* **generate**        -- produce a grounded, cited answer over graded chunks.

Every transition appends a structured entry to ``trace`` so the UI (and the
eval harness) can show exactly how the agent reasoned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .citations import (
    SourcePassage,
    build_context,
    extract_citations,
    used_passages,
)
from .config import Config, get_config
from .retriever import HybridRetriever, ScoredChunk


class SupportsChat(Protocol):
    """Minimal LLM interface the agent depends on (easy to mock)."""

    def chat(self, messages: Sequence[dict[str, str]], temperature: float = 0.0) -> str:
        ...


# --------------------------------------------------------------------- prompts
_REWRITE_SYS = (
    "You rewrite a user question into a single concise search query that "
    "maximises keyword and semantic recall for a document retriever. "
    "Reply with ONLY the rewritten query, no preamble."
)

_GRADE_SYS = (
    "You are a strict relevance grader. Given a question and a document "
    "passage, decide if the passage contains information useful to answer the "
    "question. Reply with exactly 'yes' or 'no'."
)

_REFORMULATE_SYS = (
    "Previous retrieval returned weak results. Rewrite the user's question "
    "using alternative phrasing, synonyms, or broader terms to improve recall. "
    "Reply with ONLY the new query."
)

_ANSWER_SYS = (
    "You are a precise assistant that answers ONLY from the provided sources. "
    "Every factual sentence MUST end with a citation marker copied verbatim "
    "from the sources, e.g. [doc.md p.1 / chunk 0]. If the sources do not "
    "contain the answer, say you don't know. Do not invent citations."
)


@dataclass
class TraceStep:
    """One observable step in the agent's reasoning."""

    step: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Final output of an agent run."""

    answer: str
    query: str
    rewritten_query: str
    passages: list[SourcePassage]
    cited_passages: list[SourcePassage]
    citations: list[str]
    trace: list[TraceStep]
    iterations: int


def _is_yes(text: str) -> bool:
    """Parse a grader reply into a boolean relevance verdict."""
    return text.strip().lower().startswith("y") or "yes" in text.strip().lower()[:6]


class AgenticRAG:
    """Orchestrates the analyze -> retrieve -> grade -> answer loop."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: SupportsChat,
        config: Config | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or get_config()

    # ------------------------------------------------------------- step impls
    def _rewrite_query(self, question: str) -> str:
        out = self.llm.chat(
            [
                {"role": "system", "content": _REWRITE_SYS},
                {"role": "user", "content": question},
            ]
        )
        return out.strip() or question

    def _reformulate_query(self, question: str, previous: str) -> str:
        out = self.llm.chat(
            [
                {"role": "system", "content": _REFORMULATE_SYS},
                {
                    "role": "user",
                    "content": f"Original question: {question}\n"
                    f"Previous query (weak): {previous}",
                },
            ]
        )
        return out.strip() or question

    def _grade_chunk(self, question: str, chunk_text: str) -> bool:
        out = self.llm.chat(
            [
                {"role": "system", "content": _GRADE_SYS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPassage:\n{chunk_text}",
                },
            ]
        )
        return _is_yes(out)

    def grade_chunks(
        self, question: str, scored: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]:
        """Return only the chunks the LLM grades as relevant."""
        return [sc for sc in scored if self._grade_chunk(question, sc.chunk.text)]

    def _generate_answer(
        self, question: str, passages_context: str
    ) -> str:
        if not passages_context.strip():
            return "I don't know based on the provided documents."
        out = self.llm.chat(
            [
                {"role": "system", "content": _ANSWER_SYS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSources:\n"
                    f"{passages_context}\n\nAnswer with citations:",
                },
            ]
        )
        return out.strip()

    # ------------------------------------------------------------------- loop
    def run(self, question: str, top_k: int | None = None) -> AgentResult:
        """Execute the full agentic loop and return a structured result."""
        top_k = top_k or self.config.top_k
        trace: list[TraceStep] = []

        # 1. Analyze / rewrite the query.
        query = self._rewrite_query(question)
        trace.append(
            TraceStep("analyze_query", f"Rewrote query to: {query!r}",
                      {"original": question, "rewritten": query})
        )

        relevant: list[ScoredChunk] = []
        retrieved: list[ScoredChunk] = []
        iterations = 0
        max_iters = max(1, self.config.max_agent_iterations)

        while iterations < max_iters:
            iterations += 1

            # 2. Retrieve.
            retrieved = self.retriever.retrieve(query, top_k=top_k)
            trace.append(
                TraceStep(
                    "retrieve",
                    f"Retrieved {len(retrieved)} chunks for query {query!r}.",
                    {
                        "query": query,
                        "chunk_ids": [sc.chunk.chunk_id for sc in retrieved],
                    },
                )
            )

            # 3. Self-grade relevance.
            relevant = self.grade_chunks(query, retrieved)
            trace.append(
                TraceStep(
                    "grade",
                    f"{len(relevant)}/{len(retrieved)} chunks graded relevant.",
                    {"relevant_ids": [sc.chunk.chunk_id for sc in relevant]},
                )
            )

            # 4. Decide: good enough, or reformulate and loop?
            if relevant or iterations >= max_iters:
                decision = "answer" if relevant else "answer_insufficient"
                trace.append(
                    TraceStep("decide", f"Decision: {decision}.",
                              {"iteration": iterations})
                )
                break

            new_query = self._reformulate_query(question, query)
            trace.append(
                TraceStep(
                    "reformulate",
                    f"Weak results; reformulated query to: {new_query!r}",
                    {"old_query": query, "new_query": new_query},
                )
            )
            query = new_query

        # 5. Generate grounded answer over the graded evidence.
        evidence = relevant or retrieved
        context, passages = build_context(evidence)
        answer = self._generate_answer(question, context)
        citations = extract_citations(answer)
        cited = used_passages(answer, passages)
        trace.append(
            TraceStep(
                "generate",
                f"Generated answer with {len(citations)} citation(s).",
                {"citations": citations},
            )
        )

        return AgentResult(
            answer=answer,
            query=question,
            rewritten_query=query,
            passages=passages,
            cited_passages=cited,
            citations=citations,
            trace=trace,
            iterations=iterations,
        )
