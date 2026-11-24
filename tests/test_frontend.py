"""Frontend tests using Streamlit's official AppTest harness (executes the real page code)."""
import pytest
from streamlit.testing.v1 import AppTest

from backend import services
from backend.llm.model import set_llm_override
from tests.fakes import make_llm

PAGES = ["dashboard", "assistant", "documents", "rag_explorer", "tools", "analytics", "settings"]


def page_app(name: str) -> AppTest:
    script = f"""
import sys
sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parents[1])!r})
import streamlit as st
from frontend.components import ui
from frontend.pages import {name}
ui.inject_css()
{name}.render()
"""
    return AppTest.from_string(script, default_timeout=60)


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_empty(name):
    at = page_app(name).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_with_data(name, all_docs):
    services.run_tool_manually("calculator", {"operation": "total", "numbers": [1, 2]})
    at = page_app(name).run()
    assert not at.exception, at.exception


def test_main_app_boots_and_navigates():
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "frontend" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Dashboard" in m.value for m in at.markdown)


def test_assistant_chat_roundtrip(upload):
    upload("employee_handbook.pdf")
    set_llm_override(make_llm(intent="document_qa", answer="Employees get **24 days** [1].",
                              plan=[[("document_search", {"query": "annual leave"})]]))
    at = page_app("assistant").run()
    at.chat_input[0].set_value("How many annual leave days?").run()
    assert not at.exception, at.exception
    texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert "How many annual leave days?" in texts[0] and "24 days" in texts[1]
    assert any("Sources" in e.label for e in at.expander)
    assert any("Tool executions" in e.label for e in at.expander)


def test_assistant_shows_friendly_error_without_key():
    at = page_app("assistant").run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("GROQ_API_KEY" in m.markdown[0].value for m in at.chat_message if m.markdown)


def test_tool_center_manual_run():
    at = page_app("tools").run()
    at.tabs[1].button[0].click().run()
    assert not at.exception
    assert any("success" in m.value for m in at.markdown)


def test_rag_explorer_query(upload):
    upload("employee_handbook.pdf")
    at = page_app("rag_explorer").run()
    at.text_input[0].set_value("annual leave days employees").run()
    at.button[0].click().run() if at.button else None
    assert not at.exception
