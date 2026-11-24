"""Load PDF / DOCX / TXT / CSV / XLSX files into LangChain ``Document`` objects."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

ROWS_PER_DOCUMENT = 40
_TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")


# ───────────────────────────── helpers ─────────────────────────────
def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in _TEXT_ENCODINGS:
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_csv_frame(path: Path, *, as_text: bool = False) -> pd.DataFrame:
    last_exc: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            kwargs = {"dtype": str, "keep_default_na": False} if as_text else {}
            return pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs)
        except UnicodeError as exc:
            last_exc = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DocumentProcessingError("The CSV file could not be parsed. Check that it is a valid CSV.", detail=str(exc)) from exc
    raise DocumentProcessingError("The CSV file encoding is not supported.", detail=str(last_exc))


def read_excel_frames(path: Path, *, as_text: bool = False) -> dict[str, pd.DataFrame]:
    try:
        kwargs = {"dtype": str} if as_text else {}
        frames = pd.read_excel(path, sheet_name=None, engine="openpyxl", **kwargs)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises many types
        raise DocumentProcessingError("The Excel file could not be read. It may be corrupted or password protected.", detail=str(exc)) from exc
    if as_text:
        frames = {k: v.fillna("") for k, v in frames.items()}
    return frames


def _frame_to_documents(df: pd.DataFrame, *, sheet: str | None, source_label: str) -> list[Document]:
    s = get_settings()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    n_rows = len(df)
    indexed = df.head(s.max_indexed_rows)
    cols = list(df.columns)
    header = f"{source_label} | columns: {', '.join(cols)} | total rows: {n_rows}"
    docs: list[Document] = [
        Document(
            page_content=f"DATASET OVERVIEW\n{header}\nColumn names: {', '.join(cols)}\nNumber of rows: {n_rows}",
            metadata={"sheet": sheet or "", "row_start": 0, "row_end": 0},
        )
    ]
    for start in range(0, len(indexed), ROWS_PER_DOCUMENT):
        part = indexed.iloc[start : start + ROWS_PER_DOCUMENT]
        lines = [header]
        for idx, row in part.iterrows():
            lines.append(f"Row {idx + 1}: " + "; ".join(f"{c}={row[c]}" for c in cols if str(row[c]) != ""))
        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={"sheet": sheet or "", "row_start": start + 1, "row_end": start + len(part)},
            )
        )
    if n_rows > len(indexed):
        logger.warning("%s: only first %d of %d rows indexed", source_label, len(indexed), n_rows)
    return docs


# ───────────────────────────── loaders ─────────────────────────────
def _load_pdf(path: Path) -> list[Document]:
    """Page-by-page extraction with pypdf (page numbers are 1-based)."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentProcessingError("The PDF is password protected and cannot be read.")
        total = len(reader.pages)
        docs = []
        for i, page in enumerate(reader.pages, 1):
            docs.append(Document(page_content=page.extract_text() or "", metadata={"page": i, "page_count": total}))
        return docs
    except DocumentProcessingError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise DocumentProcessingError("The PDF is corrupted or unreadable.", detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - pypdf can raise assorted errors on malformed files
        raise DocumentProcessingError("The PDF could not be read. It may be corrupted.", detail=str(exc)) from exc


def _load_docx(path: Path) -> list[Document]:
    import docx2txt

    try:
        return [Document(page_content=docx2txt.process(str(path)) or "", metadata={})]
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError("The Word document could not be read. It may be corrupted.", detail=str(exc)) from exc


def _load_txt(path: Path) -> list[Document]:
    return [Document(page_content=read_text_file(path), metadata={})]


def _load_csv(path: Path) -> list[Document]:
    df = read_csv_frame(path, as_text=True)
    return _frame_to_documents(df, sheet=None, source_label=f"CSV file {path.name}")


def _load_xlsx(path: Path) -> list[Document]:
    docs: list[Document] = []
    for name, df in read_excel_frames(path, as_text=True).items():
        if df.empty:
            continue
        docs.extend(_frame_to_documents(df, sheet=str(name), source_label=f"Excel sheet '{name}'"))
    return docs


_LOADERS = {"pdf": _load_pdf, "docx": _load_docx, "txt": _load_txt, "csv": _load_csv, "xlsx": _load_xlsx}


def load_document(path: Path, file_type: str) -> list[Document]:
    """Load a file into LangChain documents. Raises DocumentProcessingError on failure."""
    loader = _LOADERS.get(file_type)
    if loader is None:
        raise DocumentProcessingError(f"Unsupported file type: {file_type}")
    if not path.exists():
        raise DocumentProcessingError("The stored file is missing from disk.")
    docs = loader(path)
    if not any(d.page_content and d.page_content.strip() for d in docs):
        raise DocumentProcessingError(
            "No text could be extracted. The file may be empty or a scanned image (OCR is not supported)."
        )
    return docs


def load_text(path: Path, file_type: str, *, with_page_markers: bool = True) -> tuple[str, int | None]:
    """Full plain text of a document (used by summariser / comparator)."""
    docs = load_document(path, file_type)
    parts: list[str] = []
    for d in docs:
        page = d.metadata.get("page")
        marker = f"[Page {page}]\n" if (with_page_markers and page) else ""
        parts.append(marker + d.page_content)
    pages = docs[0].metadata.get("page_count") if docs else None
    return "\n\n".join(parts), pages
