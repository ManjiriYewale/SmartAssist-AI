import pytest

from backend.config.settings import get_settings
from backend.exceptions import DocumentProcessingError, UploadValidationError
from backend.rag import vector_store
from backend.rag.chunking import clean_text, split_documents
from backend.rag.document_loader import load_document
from backend.rag.embeddings import HashEmbeddings, get_embeddings
from backend.rag.ingestion import process_upload, sanitize_filename, stored_path, validate_upload
from backend.rag.retriever import format_context, retrieve
from backend.database import repository as repo


# ── loading ──
def test_load_pdf_tracks_pages(samples):
    docs = load_document(samples / "employee_handbook.pdf", "pdf")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "24 days of paid annual leave" in docs[0].page_content


@pytest.mark.parametrize("name,ftype,needle", [
    ("annual_report_2024.docx", "docx", "520 crore"),
    ("project_notes.txt", "txt", "Meera Nair"),
    ("sales_data.csv", "csv", "region"),
    ("expenses_q1.xlsx", "xlsx", "Engineering"),
])
def test_load_other_formats(samples, name, ftype, needle):
    docs = load_document(samples / name, ftype)
    assert any(needle in d.page_content for d in docs)


def test_xlsx_sheets_tracked(samples):
    docs = load_document(samples / "expenses_q1.xlsx", "xlsx")
    assert {d.metadata["sheet"] for d in docs} == {"Q1_Expenses", "Headcount"}


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(DocumentProcessingError):
        load_document(bad, "pdf")


# ── cleaning & splitting ──
def test_clean_text():
    assert clean_text("Hello\x00   world\r\n\r\n\r\n\r\nnext-\nline") == "Hello world\n\nnextline"


def test_chunk_size_and_overlap_configurable(samples):
    docs = load_document(samples / "employee_handbook.pdf", "pdf")
    small = split_documents(docs, doc_id=1, filename="h.pdf", file_type="pdf", chunk_size=200, chunk_overlap=50)
    big = split_documents(docs, doc_id=1, filename="h.pdf", file_type="pdf", chunk_size=2000, chunk_overlap=0)
    assert len(small) > len(big)
    assert all(len(c.page_content) <= 200 for c in small)
    assert small[0].metadata["page"] == 1 and small[-1].metadata["page"] == 2
    assert [c.metadata["chunk_index"] for c in small] == list(range(len(small)))


def test_overlap_must_be_smaller_than_size(samples):
    docs = load_document(samples / "project_notes.txt", "txt")
    with pytest.raises(ValueError):
        split_documents(docs, doc_id=1, filename="n", file_type="txt", chunk_size=100, chunk_overlap=100)


# ── embeddings ──
def test_embeddings_deterministic_and_normalised():
    e = get_embeddings()
    assert isinstance(e, HashEmbeddings)
    a, b = e.embed_query("annual leave policy"), e.embed_query("annual leave policy")
    assert a == b and len(a) == 384
    assert abs(sum(x * x for x in a) - 1) < 1e-6


# ── vector store + retrieval ──
def test_ingest_and_retrieve_with_sources(upload):
    rec = upload("employee_handbook.pdf")
    assert rec["chunk_count"] >= 2 and rec["page_count"] == 2
    res = retrieve("How many days of annual leave do employees get?", top_k=3)
    assert res.found
    top = res.chunks[0]
    assert top.filename == "employee_handbook.pdf" and top.page == 1
    assert "24 days" in top.content and 0 < top.score <= 1
    assert "employee_handbook.pdf, p. 1" in top.citation()


def test_retrieval_metadata_filter(all_docs):
    handbook = all_docs["employee_handbook.pdf"]["id"]
    res = retrieve("revenue profit employees", top_k=10, doc_ids=[handbook])
    assert all(c.doc_id == handbook for c in res.chunks)
    res2 = retrieve("total revenue net profit crore", top_k=10, file_type="docx")
    assert res2.chunks and all(c.file_type == "docx" for c in res2.chunks)


def test_missing_information_returns_nothing(upload):
    upload("employee_handbook.pdf")
    res = retrieve("quantum chromodynamics lattice gauge theory", top_k=5, min_score=0.3)
    assert not res.found and res.below_threshold >= 0


def test_vector_store_persists_across_reopen(upload):
    upload("project_notes.txt")
    vector_store.reset_vector_store()          # simulate application restart
    assert vector_store.collection_count() >= 1
    assert retrieve("who is the project lead").found


def test_delete_removes_vectors_and_file(upload):
    from backend.rag.ingestion import delete_document
    rec = upload("project_notes.txt")
    path = stored_path(rec["stored_name"])
    assert path.exists()
    delete_document(rec["id"])
    assert not path.exists() and vector_store.collection_count() == 0
    assert repo.list_documents() == []


def test_format_context_numbers_sources(upload):
    upload("employee_handbook.pdf")
    ctx = format_context(retrieve("sick leave", top_k=2).chunks)
    assert ctx.startswith("[1] (employee_handbook.pdf, p.")


# ── validation / security ──
def test_duplicate_upload_detected(samples):
    data = (samples / "project_notes.txt").read_bytes()
    r1, d1 = process_upload("a.txt", data)
    r2, d2 = process_upload("b.txt", data)
    assert not d1 and d2 and r1["id"] == r2["id"]
    assert len(repo.list_documents()) == 1


@pytest.mark.parametrize("name,data,msg", [
    ("malware.exe", b"MZ....", "Unsupported"),
    ("empty.txt", b"", "empty"),
    ("fake.pdf", b"hello world", "real PDF"),
    ("fake.docx", b"hello", "real DOCX"),
    ("noext", b"abc", "extension"),
])
def test_invalid_uploads_rejected(name, data, msg):
    with pytest.raises(UploadValidationError, match=msg):
        validate_upload(name, data)


def test_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from backend.config.settings import reload_settings
    reload_settings()
    with pytest.raises(UploadValidationError, match="too large"):
        validate_upload("big.txt", b"a" * (1024 * 1024 + 1))


def test_path_traversal_neutralised():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("..\\..\\win.ini.txt") == "win.ini.txt"
    with pytest.raises(UploadValidationError):
        stored_path("../escape.txt")


def test_failed_processing_is_recorded():
    rec, _ = process_upload("scan.pdf", b"%PDF-1.4 garbage that is not parseable")
    assert rec["status"] == "failed" and rec["error_message"]
    assert vector_store.collection_count() == 0


def test_empty_text_file_marked_failed():
    rec, _ = process_upload("blank.txt", b"   \n  \n")
    assert rec["status"] == "failed"
