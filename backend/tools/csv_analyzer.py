"""TOOL 3 – CSV / Excel analysis using predefined pandas operations (no user code is executed)."""
from __future__ import annotations

import re
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.config.settings import TABULAR_TYPES
from backend.exceptions import DocumentProcessingError, ToolInputError
from backend.rag.document_loader import read_csv_frame, read_excel_frames
from backend.rag.ingestion import stored_path
from backend.tools.common import py, resolve_document, safe_tool_call

MAX_ROWS_RETURNED = 25
MAX_COLS_RETURNED = 15


def load_dataframe(doc: dict, sheet: str | None = None) -> tuple[pd.DataFrame, str | None]:
    path = stored_path(doc["stored_name"])
    if doc["file_type"] == "csv":
        df = read_csv_frame(path)
        used_sheet = None
    else:
        frames = read_excel_frames(path)
        if not frames:
            raise DocumentProcessingError("The workbook has no sheets.")
        if sheet:
            match = [k for k in frames if str(k).lower() == sheet.lower()]
            if not match:
                raise ToolInputError(f"Sheet '{sheet}' not found. Sheets: {', '.join(map(str, frames))}")
            used_sheet = str(match[0])
        else:
            used_sheet = str(next(iter(frames)))
        df = frames[used_sheet if used_sheet in frames else next(iter(frames))]
    df.columns = [str(c).strip() for c in df.columns]
    return df, used_sheet


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_column(df: pd.DataFrame, name: str | None, *, required: bool = True) -> str | None:
    if not name:
        if required:
            raise ToolInputError(f"A column name is required. Columns: {', '.join(df.columns)}")
        return None
    cols = list(df.columns)
    if name in cols:
        return name
    low = [c for c in cols if c.lower() == name.lower()]
    if len(low) == 1:
        return low[0]
    n = _norm(name)
    exact = [c for c in cols if _norm(c) == n]
    if len(exact) == 1:
        return exact[0]
    part = [c for c in cols if n and (n in _norm(c) or _norm(c) in n)]
    if len(part) == 1:
        return part[0]
    raise ToolInputError(f"Column '{name}' not found" + (f" or ambiguous ({', '.join(part)})" if part else "") +
                         f". Columns: {', '.join(cols)}")


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(r"[,\s₹$€£%]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        s = to_numeric(df[c])
        if s.notna().sum() >= max(1, int(0.6 * df[c].notna().sum())) and df[c].notna().sum() > 0:
            out.append(c)
    return out


def _rows(df: pd.DataFrame, n: int = MAX_ROWS_RETURNED) -> list[dict]:
    sub = df.iloc[:, :MAX_COLS_RETURNED].head(n)
    return [{k: py(v) for k, v in rec.items()} for rec in sub.to_dict("records")]


def run_csv_analysis(document: str | None, operation: str, column: str | None = None, group_by: str | None = None,
                     aggregation: str = "sum", top_n: int = 5, sheet: str | None = None) -> dict:
    doc = resolve_document(document, file_types=TABULAR_TYPES, what="CSV/Excel file")
    df, used_sheet = load_dataframe(doc, sheet)
    base = {"document": doc["filename"], "sheet": used_sheet, "operation": operation, "rows": int(len(df)),
            "columns_count": int(df.shape[1])}
    if df.empty:
        raise ToolInputError("The dataset has no rows.")
    top_n = max(1, min(int(top_n or 5), 50))

    if operation == "overview":
        num = _numeric_columns(df)
        return {**base, "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                                     "numeric": c in num} for c in df.columns],
                "numeric_columns": num, "preview": _rows(df, 5)}

    if operation == "describe":
        num = _numeric_columns(df)
        if not num:
            raise ToolInputError("No numeric columns found to describe.")
        desc = pd.DataFrame({c: to_numeric(df[c]) for c in num}).describe().round(4)
        return {**base, "statistics": py(desc.to_dict())}

    if operation == "missing_values":
        miss = df.isna().sum()
        blank = df.apply(lambda s: s.astype(str).str.strip().eq("").sum() if s.dtype == object or str(s.dtype) == "str" else 0)
        total = (miss + blank)
        return {**base, "missing_per_column": {c: int(total[c]) for c in df.columns if total[c] > 0},
                "total_missing_cells": int(total.sum()), "rows_with_missing": int((df.isna().any(axis=1)).sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "percent_missing_per_column": {c: round(float(total[c]) / len(df) * 100, 2) for c in df.columns if total[c] > 0}}

    if operation in {"sum", "mean", "median", "min", "max", "count"}:
        if operation == "count" and not column:
            return {**base, "result": int(len(df))}
        cols = [resolve_column(df, column)] if column else _numeric_columns(df)
        if not cols:
            raise ToolInputError("No numeric columns found.")
        results, details = {}, {}
        for c in cols:
            s = to_numeric(df[c])
            if operation == "count":
                results[c] = int(df[c].notna().sum())
                continue
            valid = s.dropna()
            if valid.empty:
                raise ToolInputError(f"Column '{c}' has no numeric values.")
            results[c] = py(getattr(valid, operation)())
            if operation in {"min", "max"}:
                idx = valid.idxmax() if operation == "max" else valid.idxmin()
                details[c] = {"row_number": int(df.index.get_loc(idx)) + 1,
                              "row": {k: py(v) for k, v in df.loc[idx].to_dict().items()}}
            details.setdefault(c, {})["values_used"] = int(valid.shape[0])
            skipped = int(s.isna().sum() - df[c].isna().sum())
            if skipped:
                details[c]["non_numeric_ignored"] = skipped
            details[c]["empty_ignored"] = int(df[c].isna().sum())
        out = {**base, "result": results[cols[0]] if len(cols) == 1 else results, "details": details}
        return out

    if operation == "unique_values":
        c = resolve_column(df, column)
        vals = df[c].dropna().unique()
        return {**base, "column": c, "unique_count": int(len(vals)), "values": [py(v) for v in vals[:30]]}

    if operation == "value_counts":
        c = resolve_column(df, column)
        vc = df[c].value_counts(dropna=False).head(top_n)
        return {**base, "column": c, "counts": {str(k): int(v) for k, v in vc.items()}}

    if operation == "top_n":
        c = resolve_column(df, column)
        s = to_numeric(df[c])
        order = s.sort_values(ascending=False).dropna().head(top_n).index
        return {**base, "column": c, "top_rows": _rows(df.loc[order], top_n)}

    if operation == "group_by":
        g = resolve_column(df, group_by)
        agg = aggregation if aggregation in {"sum", "mean", "count", "min", "max"} else "sum"
        c = resolve_column(df, column, required=(agg != "count"))
        if agg == "count" and not c:
            res = df.groupby(g, dropna=False).size().sort_values(ascending=False)
        else:
            tmp = pd.DataFrame({g: df[g], c: to_numeric(df[c])})
            res = getattr(tmp.groupby(g, dropna=False)[c], agg)().sort_values(ascending=False)
        return {**base, "group_by": g, "column": c, "aggregation": agg,
                "groups": {str(k): py(v) for k, v in res.head(50).items()}}

    raise ToolInputError(f"Unknown operation '{operation}'.")


class CsvAnalyzerInput(BaseModel):
    document: Optional[str] = Field(None, description="CSV/XLSX file name or id. Optional if only one dataset exists.")
    operation: Literal["overview", "describe", "missing_values", "sum", "mean", "median", "min", "max", "count",
                       "unique_values", "value_counts", "top_n", "group_by"] = Field(
        description="overview: shape/columns/preview. describe: statistics of numeric columns. missing_values: null counts. "
                    "sum/mean/median/min/max/count: of 'column' (all numeric columns if omitted; min/max also return the row). "
                    "unique_values/value_counts: of 'column'. top_n: rows with the highest 'column'. "
                    "group_by: aggregate 'column' per 'group_by' category using 'aggregation'.")
    column: Optional[str] = Field(None, description="Column name (case-insensitive, fuzzy match).")
    group_by: Optional[str] = Field(None, description="Category column for the group_by operation.")
    aggregation: Literal["sum", "mean", "count", "min", "max"] = Field("sum", description="Aggregation for group_by.")
    top_n: int = Field(5, ge=1, le=50, description="Rows to return for top_n / value_counts.")
    sheet: Optional[str] = Field(None, description="Sheet name for Excel files (default: first sheet).")


@tool("csv_excel_analyzer", args_schema=CsvAnalyzerInput)
def csv_excel_analyzer(document: Optional[str] = None, operation: str = "overview", column: Optional[str] = None,
                       group_by: Optional[str] = None, aggregation: str = "sum", top_n: int = 5,
                       sheet: Optional[str] = None) -> str:
    """Exact statistics on an uploaded CSV or Excel file: row/column counts, sums, averages, minimum/maximum
    (with the matching row), missing values, group totals and top-N rows. Use this for numeric questions about datasets."""
    return safe_tool_call("csv_excel_analyzer",
                          lambda: run_csv_analysis(document, operation, column, group_by, aggregation, top_n, sheet))
