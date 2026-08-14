import pytest
from pathlib import Path
from ingestion.chunk_and_embed import CORE_BOOKS as CHUNK_CORE_BOOKS, load_documents as load_chunk_docs
from ingestion.build_bm25_index import CORE_BOOKS as BM25_CORE_BOOKS, load_documents as load_bm25_docs


def test_core_books_limit():

    assert len(CHUNK_CORE_BOOKS) == 5, f"Expected 5 books in chunk_and_embed CORE_BOOKS, got {len(CHUNK_CORE_BOOKS)}"
    assert len(BM25_CORE_BOOKS) == 5, f"Expected 5 books in build_bm25_index CORE_BOOKS, got {len(BM25_CORE_BOOKS)}"

    expected = {f"book-{i}.pdf" for i in range(1, 6)}
    assert CHUNK_CORE_BOOKS == expected
    assert BM25_CORE_BOOKS == expected


def test_load_documents_filters_excess_books(tmp_path, monkeypatch):
    """Verify that PDFs exceeding book-5.pdf (e.g. book-6.pdf) are excluded."""
    for i in range(1, 8):
        (tmp_path / f"book-{i}.pdf").write_text(f"Content for book {i}", encoding="utf-8")

    monkeypatch.setattr("ingestion.chunk_and_embed.read_file_content", lambda p: "sample text")
    monkeypatch.setattr("ingestion.build_bm25_index.read_file_content", lambda p: "sample text")

    chunk_docs = load_chunk_docs(str(tmp_path))
    bm25_docs = load_bm25_docs(str(tmp_path))

    loaded_sources_chunk = {d["source"].lower() for d in chunk_docs}
    loaded_sources_bm25 = {d["source"].lower() for d in bm25_docs}

    expected_sources = {f"book-{i}.pdf" for i in range(1, 6)}

    assert loaded_sources_chunk == expected_sources
    assert loaded_sources_bm25 == expected_sources
    assert "book-6.pdf" not in loaded_sources_chunk
    assert "book-7.pdf" not in loaded_sources_chunk
