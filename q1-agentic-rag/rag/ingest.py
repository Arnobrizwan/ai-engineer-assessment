"""Document loading + ingestion into :class:`~rag.chunking.Chunk` objects.

Supports ``.txt``, ``.md`` and ``.pdf``. PDFs are read page by page (via
``pypdf``) so citation page numbers are accurate; text/markdown are treated as
a single page. The output is a flat list of chunks ready for indexing by the
retriever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .chunking import Chunk, chunk_text
from .config import Config, get_config

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_text_file(path: Path) -> list[tuple[int, str]]:
    """Return ``[(page, text)]`` for a plain-text/markdown file (single page)."""
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def _read_pdf_file(path: Path) -> list[tuple[int, str]]:
    """Return ``[(page, text), ...]`` for each page of a PDF."""
    from pypdf import PdfReader  # lazy import; only needed for PDFs

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append((i, page.extract_text() or ""))
    return pages


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Load a single document as a list of ``(page_number, text)`` tuples."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_file(path)
    if suffix in {".txt", ".md"}:
        return _read_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_path(path: Path, config: Config | None = None) -> list[Chunk]:
    """Load and chunk a single file into a list of chunks."""
    config = config or get_config()
    pages = load_pages(path)
    chunks: list[Chunk] = []
    running_index = 0
    for page_no, text in pages:
        page_chunks = chunk_text(
            text,
            doc_name=path.name,
            page=page_no,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            start_index=running_index,
        )
        running_index += len(page_chunks)
        chunks.extend(page_chunks)
    return chunks


def ingest_paths(
    paths: Iterable[Path], config: Config | None = None
) -> list[Chunk]:
    """Ingest many files, concatenating their chunks."""
    config = config or get_config()
    all_chunks: list[Chunk] = []
    for path in paths:
        all_chunks.extend(ingest_path(Path(path), config))
    return all_chunks


def ingest_directory(
    directory: Path | None = None, config: Config | None = None
) -> list[Chunk]:
    """Ingest every supported document under ``directory`` (default: data dir)."""
    config = config or get_config()
    directory = Path(directory) if directory else config.data_dir
    paths = sorted(
        p for p in directory.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return ingest_paths(paths, config)
