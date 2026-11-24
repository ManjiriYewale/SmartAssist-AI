import json

import pandas as pd
import pytest

from backend.llm.model import set_llm_override
from backend.tools import ALL_TOOLS, TOOLS_BY_NAME, tool_catalog
from backend.tools.calculator import calculator, safe_eval
from backend.tools.comparator import document_comparator, term_overlap
from backend.tools.csv_analyzer import csv_excel_analyzer
from backend.tools.document_manager import document_manager
from backend.tools.document_search import document_search
from backend.tools.summarizer import condense_text, document_summarizer
from tests.fakes import make_llm


def call(tool, **args):
    return json.loads(tool.invoke(args))


# ── registry ──
def test_six_real_langchain_tools():
    assert {t.name for t in ALL_TOOLS} == {"calculator", "document_search", "csv_excel_analyzer",
                                           "document_summarizer", "document_comparator", "document_manager"}
    for t in ALL_TOOLS:
        assert t.description and t.args_schema is not None
    assert len(tool_catalog()) == 6


# ── calculator ──
def test_calculator_arithmetic_and_percent():
    assert call(calculator, operation="evaluate", expression="(1200*3)+450/2")["result"] == 3825
    assert call(calculator, operation="evaluate", expression="2^10")["result"] == 1024
    assert call(calculator, operation="percentage_of", amount=2000, percent=15)["result"] == 300
    assert call(calculator, operation="percentage_change", amount=410, other=520)["result"] == pytest.approx(26.8293, 1e-4)
    assert call(calculator, operation="percentage_share", amount=25, other=200)["result"] == 12.5


def test_calculator_gst():
    r = call(calculator, operation="add_gst", amount=2500, percent=18)
    assert (r["gst_amount"], r["total_amount"], r["cgst"]) == (450.0, 2950.0, 225.0)
    r = call(calculator, operation="remove_gst", amount=2950, percent=18)
    assert (r["net_amount"], r["gst_amount"]) == (2500.0, 450.0)
    assert call(calculator, operation="add_gst", amount=1000)["gst_rate_percent"] == 18.0  # default rate


def test_calculator_total_average():
    r = call(calculator, operation="average", numbers=[10, 20, 60])
    assert r["average"] == 30 and r["total"] == 90 and r["maximum"] == 60
    assert call(calculator, operation="total", numbers=[1.5, 2.5])["result"] == 4


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "().__class__", "1/0", "10**999",
                                  "abs(-1)", "a+1", "", "x" * 300])
def test_calculator_rejects_unsafe_or_invalid(expr):
    r = call(calculator, operation="evaluate", expression=expr)
    assert r["status"] == "error"


def test_calculator_missing_args_and_bad_operation():
    assert call(calculator, operation="add_gst")["status"] == "error"
    with pytest.raises(Exception):
        calculator.invoke({"operation": "rm -rf"})  # fails pydantic validation, never executes
    assert safe_eval("5% + 1") == pytest.approx(1.05)


# ── CSV / Excel ──
def test_csv_overview_and_missing(upload):
    upload("sales_data.csv")
    ov = call(csv_excel_analyzer, operation="overview")
    assert ov["rows"] == 60 and "revenue" in [c["name"] for c in ov["columns"]]
    miss = call(csv_excel_analyzer, operation="missing_values")
    assert miss["missing_per_column"] == {"region": 2, "units": 1}


def test_csv_aggregates_match_pandas(upload, samples):
    upload("sales_data.csv")
    df = pd.read_csv(samples / "sales_data.csv")
    assert call(csv_excel_analyzer, operation="sum", column="revenue")["result"] == df["revenue"].sum()
    assert call(csv_excel_analyzer, operation="mean", column="Revenue")["result"] == pytest.approx(df["revenue"].mean())
    mx = call(csv_excel_analyzer, operation="max", column="revenue")
    assert mx["result"] == df["revenue"].max()
    assert mx["details"]["revenue"]["row"]["revenue"] == df["revenue"].max()   # returns the matching row
    assert call(csv_excel_analyzer, operation="min", column="units")["result"] == df["units"].min()
    g = call(csv_excel_analyzer, operation="group_by", group_by="product", column="revenue", aggregation="sum")
    assert g["groups"]["Laptop"] == df[df["product"] == "Laptop"]["revenue"].sum()
    top = call(csv_excel_analyzer, operation="top_n", column="revenue", top_n=3)
    assert [r["revenue"] for r in top["top_rows"]] == sorted(df["revenue"], reverse=True)[:3]


def test_excel_sheets_and_errors(upload):
    upload("expenses_q1.xlsx")
    assert call(csv_excel_analyzer, operation="sum", column="amount")["result"] == 824000
    hc = call(csv_excel_analyzer, operation="sum", column="employees", sheet="Headcount")
    assert hc["result"] == 227 and hc["sheet"] == "Headcount"
    assert call(csv_excel_analyzer, operation="sum", column="nonexistent")["status"] == "error"
    assert call(csv_excel_analyzer, operation="sum", column="amount", sheet="Nope")["status"] == "error"


def test_csv_requires_document_when_ambiguous(upload):
    upload("sales_data.csv"); upload("expenses_q1.xlsx")
    r = call(csv_excel_analyzer, operation="overview")
    assert r["status"] == "error" and "specify" in r["error"].lower()
    assert call(csv_excel_analyzer, document="sales_data.csv", operation="overview")["status"] == "ok"


def test_csv_tool_rejects_non_tabular(upload):
    upload("project_notes.txt")
    assert call(csv_excel_analyzer, document="project_notes.txt", operation="overview")["status"] == "error"


# ── document search ──
def test_document_search_returns_sources_and_artifact(upload):
    upload("employee_handbook.pdf")
    msg = document_search.invoke({"name": "document_search", "id": "1", "type": "tool_call",
                                 "args": {"query": "maternity leave weeks"}})
    data = json.loads(msg.content)
    assert data["found"] and data["results"][0]["page"] == 1
    assert msg.artifact and msg.artifact[0]["filename"] == "employee_handbook.pdf"


def test_document_search_scoped_and_empty(upload):
    upload("employee_handbook.pdf"); upload("project_notes.txt")
    r = call(document_search, query="project lead", document="project_notes.txt")
    assert {x["document"] for x in r["results"]} == {"project_notes.txt"}
    assert call(document_search, query="project lead", document="ghost.pdf")["status"] == "error"


# ── summariser (LLM scripted) ──
def test_summarizer_short_document_single_call(upload):
    upload("project_notes.txt")
    llm = make_llm(); set_llm_override(llm)
    r = call(document_summarizer, document="project_notes", mode="key_points")
    assert r["status"] == "ok" and r["mode"] == "key_points" and r["coverage"] == "full"
    final_prompt = [t for role, t in llm.log if role == "summary"][0]
    assert "KEY POINTS" in final_prompt and "Meera Nair" in final_prompt


def test_summarizer_map_reduce_for_long_text():
    llm = make_llm(); set_llm_override(llm)
    text = ("Paragraph about topic. " * 200 + "\n\n") * 8        # ≈ 36k chars
    out, info = condense_text(text, "big.txt")
    assert info["sections"] > 1 and out.count("condensed notes") == info["sections"]
    assert sum(1 for r, _ in llm.log if r == "map") == info["sections"]


def test_summarizer_llm_failure_is_reported(upload):
    upload("project_notes.txt")
    set_llm_override(make_llm(fail_on="summary"))
    r = call(document_summarizer, document="project_notes")
    assert r["status"] == "error" and "rate-limited" in r["error"]


# ── comparator ──
def test_comparator_metrics_and_report(upload):
    upload("annual_report_2023.docx"); upload("annual_report_2024.docx")
    llm = make_llm(); set_llm_override(llm)
    r = call(document_comparator, document_a="annual_report_2023.docx", document_b="annual_report_2024.docx",
             focus="revenue")
    assert r["status"] == "ok" and "[compare]" in r["report"]
    assert 0 < r["metrics"]["vocabulary_jaccard_similarity"] < 1
    prompt = [t for role, t in llm.log if role == "compare"][0]
    assert "410 crore" in prompt and "520 crore" in prompt and "Focus especially on: revenue" in prompt


def test_comparator_validation(upload):
    upload("annual_report_2023.docx")
    set_llm_override(make_llm())
    assert call(document_comparator, document_a="annual_report_2023.docx", document_b="annual_report_2023.docx")["status"] == "error"
    assert call(document_comparator, document_a="annual_report_2023.docx", document_b="missing.docx")["status"] == "error"


def test_comparator_tabular_schema_diff(upload):
    upload("sales_data.csv"); upload("expenses_q1.xlsx")
    set_llm_override(make_llm())
    r = call(document_comparator, document_a="sales_data.csv", document_b="expenses_q1.xlsx")
    assert "column_comparison" in r["metrics"]
    assert r["metrics"]["column_comparison"]["only_in_a"] == sorted(["date", "region", "product", "units", "unit_price", "revenue"])


def test_term_overlap_identical_is_one():
    assert term_overlap("alpha beta gamma", "alpha beta gamma")["vocabulary_jaccard_similarity"] == 1.0


# ── document manager ──
def test_document_manager_actions(upload):
    upload("project_notes.txt"); upload("sales_data.csv")
    lst = call(document_manager, action="list")
    assert lst["count"] == 2
    assert call(document_manager, action="list", file_type="csv")["count"] == 1
    assert call(document_manager, action="available")["count"] == 2
    md = call(document_manager, action="metadata", document="sales_data.csv")
    assert md["document"]["type"] == "csv" and md["document"]["chunks"] > 0
    assert call(document_manager, action="status", document="project_notes")["status"] == "processed"
    assert call(document_manager, action="metadata")["status"] == "error"
