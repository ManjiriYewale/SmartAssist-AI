"""Agent tests: the REAL LangGraph workflow, real tools and RAG, with a scripted chat model."""
import json

from langchain_core.messages import AIMessage

from backend.agents.routing import route_after_agent, route_after_intent, route_after_tools
from backend.agents.graph import run_agent
from backend.chains.response_chain import parse_intent
from backend.config.settings import get_settings
from backend.database import repository as repo
from backend.llm.model import set_llm_override
from tests.fakes import make_llm


def roles(llm):
    return [r for r, _ in llm.log]


# ── intent parsing ──
def test_parse_intent_variants():
    assert parse_intent('{"intent": "comparison", "reason": "x"}')[0] == "comparison"
    assert parse_intent('```json\n{"intent":"data_analysis"}\n```')[0] == "data_analysis"
    assert parse_intent("The intent is calculation.")[0] == "calculation"
    assert parse_intent("blah blah")[0] is None
    assert parse_intent('{"intent": "hack_the_planet"}')[0] is None


# ── routing functions ──
def test_routing_functions():
    call = lambda n: AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": "1", "type": "tool_call"}])
    assert route_after_intent({"intent": "general_chat"}) == "general"
    assert route_after_intent({"intent": "comparison"}) == "agent"
    assert route_after_intent({"status": "error"}) == "error"
    assert route_after_agent({"messages": [call("document_search")], "steps": 0}) == "rag"
    assert route_after_agent({"messages": [call("calculator")], "steps": 0}) == "tools"
    assert route_after_agent({"messages": [call("calculator")], "steps": 99}) == "respond"          # execution limit
    assert route_after_agent({"messages": [AIMessage(content="READY")], "intent": "document_qa", "retrieval_done": False}) == "rag"
    assert route_after_agent({"messages": [AIMessage(content="READY")], "intent": "document_qa", "retrieval_done": True}) == "respond"
    assert route_after_agent({"messages": [AIMessage(content="READY")], "intent": "calculation"}) == "respond"
    assert route_after_tools({"steps": 1}) == "agent"
    assert route_after_tools({"steps": 99}) == "respond"
    assert route_after_tools({"selected_action": "rag_fallback", "steps": 1}) == "respond"


# ── end-to-end graph runs ──
def test_document_qa_uses_rag_tool_and_cites_sources(upload):
    upload("employee_handbook.pdf")
    llm = make_llm(intent="document_qa", answer="Employees get 24 days [1].",
                   plan=[[("document_search", {"query": "annual leave days"})]])
    set_llm_override(llm)
    st = run_agent("How many days of annual leave do I get?")
    assert st["trace"] == ["query_understanding", "intent_classification", "agent_decision", "rag_retrieval",
                           "agent_decision", "response_generation", "final_response"]
    assert st["status"] == "completed" and st["selected_action"] == "rag" and st["intent"] == "document_qa"
    assert st["sources"][0]["filename"] == "employee_handbook.pdf" and st["sources"][0]["page"] == 1
    response_prompt = [t for r, t in llm.log if r == "response"][0]
    assert "24 days of paid annual leave" in response_prompt              # retrieved context reached the LLM
    assert st["final_response"] == "Employees get 24 days [1]."
    assert [o["tool"] for o in st["tool_outputs"]] == ["document_search"]


def test_fallback_retrieval_when_agent_skips_tools(upload):
    upload("employee_handbook.pdf")
    llm = make_llm(intent="document_qa", plan=[])           # agent immediately says READY
    set_llm_override(llm)
    st = run_agent("What is the maternity leave policy?")
    assert "rag_retrieval" in st["trace"] and st["retrieval_done"]
    assert st["sources"] and st["tool_calls"][0]["name"] == "document_search"


def test_multi_tool_execution(upload):
    upload("sales_data.csv")
    llm = make_llm(intent="data_analysis", answer="done", plan=[
        [("csv_excel_analyzer", {"document": "sales_data.csv", "operation": "sum", "column": "revenue"})],
        [("calculator", {"operation": "add_gst", "amount": 1000, "percent": 18})],
    ])
    set_llm_override(llm)
    st = run_agent("Total revenue of sales_data and GST on 1000?")
    assert [o["tool"] for o in st["tool_outputs"]] == ["csv_excel_analyzer", "calculator"]
    assert all(o["status"] == "success" for o in st["tool_outputs"])
    assert st["selected_action"] == "tools" and st["steps"] == 2
    tool_prompt = [t for r, t in llm.log if r == "response"][0]
    assert "total_amount" in tool_prompt and "1180" in tool_prompt
    assert len(repo.list_tool_executions()) == 2            # logged to the database


def test_parallel_tool_calls_in_one_step(upload):
    upload("sales_data.csv")
    llm = make_llm(intent="data_analysis", plan=[[
        ("csv_excel_analyzer", {"operation": "overview"}), ("calculator", {"operation": "total", "numbers": [1, 2]})]])
    set_llm_override(llm)
    st = run_agent("overview and total")
    assert len(st["tool_outputs"]) == 2 and st["steps"] == 1


def test_general_chat_skips_agent_and_tools():
    llm = make_llm(intent="general_chat", answer="Hello! I can help with your documents.")
    set_llm_override(llm)
    st = run_agent("Hi there")
    assert st["trace"] == ["query_understanding", "intent_classification", "response_generation", "final_response"]
    assert "agent" not in roles(llm) and st["tool_calls"] == []
    assert st["selected_action"] == "general" and st["final_response"].startswith("Hello")


def test_missing_information_never_hallucinates(upload):
    upload("project_notes.txt")
    llm = make_llm(intent="document_qa", answer="HALLUCINATED ANSWER",
                   plan=[[("document_search", {"query": "quantum chromodynamics lattice gauge theory"})]])
    set_llm_override(llm)
    st = run_agent("Explain quantum chromodynamics lattice gauge theory")
    assert st["sources"] == []
    assert "HALLUCINATED" not in st["final_response"] and "couldn't find" in st["final_response"]
    assert "response" not in roles(llm)                     # LLM was not even asked to answer


def test_no_documents_message():
    set_llm_override(make_llm(intent="document_qa"))
    st = run_agent("What does the contract say?")
    assert "don't have any processed documents" in st["final_response"]


def test_step_limit_prevents_infinite_loop(upload, monkeypatch):
    upload("sales_data.csv")
    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    from backend.config.settings import reload_settings
    reload_settings()
    # a different call each round, so de-duplication cannot hide the loop
    llm = make_llm(intent="data_analysis",
                   always_call=lambda i: ("calculator", {"operation": "evaluate", "expression": f"{i}+1"}))
    set_llm_override(llm)
    st = run_agent("loop forever")
    assert st["steps"] == 3 and st["status"] == "step_limit"
    assert "stopped after 3" in st["final_response"]


def test_identical_repeated_calls_are_deduplicated(upload):
    llm = make_llm(intent="calculation", plan=[[("calculator", {"operation": "total", "numbers": [1, 2]})],
                                               [("calculator", {"operation": "total", "numbers": [1, 2]})]])
    set_llm_override(llm)
    st = run_agent("sum 1 2")
    assert len(repo.list_tool_executions()) == 1
    assert st["tool_outputs"][0]["tool"] == "calculator" and len(st["tool_outputs"]) == 1


def test_unknown_tool_and_invalid_args_are_contained(upload):
    llm = make_llm(intent="calculation", plan=[
        [("delete_everything", {"x": 1})],
        [("calculator", {"operation": "not_real"})],
    ])
    set_llm_override(llm)
    st = run_agent("do weird things")
    assert [o["status"] for o in st["tool_outputs"]] == ["error", "error"]
    assert st["status"] == "completed"                      # the workflow survives bad tool calls
    logs = repo.list_tool_executions()
    assert {l["status"] for l in logs} == {"error"}


def test_llm_failure_gives_friendly_error(upload):
    upload("project_notes.txt")
    set_llm_override(make_llm(intent="document_qa", fail_on="response", plan=[[("document_search", {"query": "project lead"})]]))
    st = run_agent("Who is the project lead?")
    assert st["status"] == "error" and "rate-limited" in st["final_response"]
    assert "Traceback" not in st["final_response"] and "RuntimeError" not in st["final_response"]


def test_classifier_failure_falls_back_to_document_qa(upload):
    upload("project_notes.txt")
    set_llm_override(make_llm(fail_on="classify", answer="Meera Nair [1]"))
    st = run_agent("Who is the project lead?")
    assert st["intent"] == "document_qa" and st["status"] == "completed" and st["sources"]
    assert any("fell back" in w for w in st["warnings"])


def test_missing_api_key_message():
    st = run_agent("hello")                                  # no override, GROQ_API_KEY empty
    assert st["status"] == "error" and "GROQ_API_KEY" in st["final_response"]


def test_followup_is_rewritten_using_history(upload):
    upload("employee_handbook.pdf")
    llm = make_llm(intent="document_qa", plan=[])
    set_llm_override(llm)
    hist = [{"role": "user", "content": "Tell me about sick leave"}, {"role": "assistant", "content": "12 days."}]
    st = run_agent("and for maternity?", history=hist)
    assert st["rewritten_query"].startswith("REWRITTEN:") and "rewrite" in roles(llm)


def test_empty_query_rejected():
    st = run_agent("   ")
    assert st["status"] == "error" and "enter a question" in st["final_response"].lower()


def test_services_ask_persists_everything(upload):
    from backend import services
    upload("employee_handbook.pdf")
    set_llm_override(make_llm(intent="document_qa", answer="24 days [1]", plan=[[("document_search", {"query": "annual leave"})]]))
    r1 = services.ask("How many annual leave days?")
    cid = r1["conversation_id"]
    set_llm_override(make_llm(intent="general_chat", answer="You're welcome"))
    r2 = services.ask("thanks", cid)
    msgs = repo.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["sources"][0]["filename"] == "employee_handbook.pdf" and msgs[1]["tool_calls"][0]["name"] == "document_search"
    q = repo.list_queries()
    assert len(q) == 2 and q[1]["intent"] == "document_qa" and q[1]["tools_used"] == ["document_search"]
    assert repo.get_conversation(cid)["title"].startswith("How many annual")
