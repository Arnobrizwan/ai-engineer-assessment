"""Streamlit UI for the Agentic RAG system.

Features
--------
* Use the bundled sample documents or upload your own PDF/TXT/MD files.
* Ask a question and get a grounded, cited answer.
* Inspect the agent's full reasoning trace (analyze -> retrieve -> grade ->
  reformulate -> answer).
* See the exact source passages backing the answer, each with its citation.

Run with:  ``streamlit run app.py``
(Requires a local Ollama server with ``llama3.1:8b`` and ``nomic-embed-text``.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag.config import get_config
from rag.ingest import ingest_directory, ingest_paths
from rag.llm import LLMClient
from rag.pipeline import build_agent

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")
CONFIG = get_config()


@st.cache_resource(show_spinner=False)
def _get_llm() -> LLMClient:
    return LLMClient(CONFIG)


def _build_agent_for_session(uploaded_files):
    """Ingest the chosen documents and build a cached agent for them."""
    llm = _get_llm()
    if uploaded_files:
        tmpdir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
        paths = []
        for uf in uploaded_files:
            dest = tmpdir / uf.name
            dest.write_bytes(uf.getbuffer())
            paths.append(dest)
        chunks = ingest_paths(paths, CONFIG)
        source_label = ", ".join(p.name for p in paths)
    else:
        chunks = ingest_directory(config=CONFIG)
        source_label = "bundled sample documents"
    agent = build_agent(chunks, llm=llm, embed_fn=llm.embed, config=CONFIG)
    return agent, len(chunks), source_label


def main() -> None:
    st.title("🔎 Agentic RAG")
    st.caption(
        "Hybrid retrieval (dense + BM25 + RRF + rerank) driving a self-grading "
        "agent loop with grounded citations."
    )

    llm = _get_llm()
    with st.sidebar:
        st.header("Setup")
        online = llm.is_available()
        st.write("Ollama:", "🟢 reachable" if online else "🔴 not reachable")
        st.write(f"LLM: `{CONFIG.llm_model}`")
        st.write(f"Embeddings: `{CONFIG.embed_model}`")
        st.divider()
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload PDF / TXT / MD (or leave empty for samples)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        top_k = st.slider("Chunks to retrieve (top-k)", 2, 10, CONFIG.top_k)

    if not online:
        st.warning(
            "Ollama is not reachable. Start it and pull the models:\n\n"
            "`ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`."
        )

    agent, n_chunks, source_label = _build_agent_for_session(uploaded)
    st.info(f"Indexed **{n_chunks}** chunks from {source_label}.")

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. How many moons does Mars have and what are they called?",
    )
    ask = st.button("Ask", type="primary", disabled=not question.strip())

    if ask and question.strip():
        with st.spinner("Agent reasoning..."):
            result = agent.run(question, top_k=top_k)

        st.subheader("Answer")
        st.markdown(result.answer)

        if result.citations:
            st.caption("Citations: " + "  ".join(f"`{c}`" for c in result.citations))

        cited = result.cited_passages or result.passages
        with st.expander(f"📄 Source passages ({len(cited)})", expanded=True):
            for p in cited:
                st.markdown(f"**{p.citation}**")
                st.write(p.text)
                st.divider()

        with st.expander("🧠 Agent reasoning trace", expanded=False):
            for i, step in enumerate(result.trace, start=1):
                st.markdown(f"**{i}. {step.step}** — {step.detail}")
                if step.data:
                    st.json(step.data, expanded=False)
            st.caption(
                f"Rewritten query: `{result.rewritten_query}` · "
                f"iterations: {result.iterations}"
            )


if __name__ == "__main__":
    main()
