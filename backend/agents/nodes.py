"""LangGraph nodes.

query_understanding → intent_classification → agent_decision → (rag_retrieval | tool_execution)* →
response_generation → final_response
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agents.state import AgentState
from backend.chains.prompts import AGENT_SYSTEM_PROMPT, INTENTS
from backend.chains.response_chain import classify_intent, generate_answer, history_to_messages, rewrite_query
from backend.config.settings import get_settings
from backend.database import repository as repo
from backend.exceptions import LLMConfigurationError, LLMServiceError, LLMToolCallError, SmartAssistError
from backend.llm.model import get_llm, invoke_llm
from backend.rag.retriever import RetrievedChunk, format_context
from backend.tools import ALL_TOOLS, TOOLS_BY_NAME
from backend.tools.common import truncate
from backend.tools.document_manager import library_overview

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 4000
MAX_CONTEXT_CHARS = 12000
MAX_CHUNKS_KEPT = 12


def _trace(state: AgentState, name: str) -> list[str]:
    return (state.get("trace") or []) + [name]


# ───────────────────────── 1. query understanding ─────────────────────────
def query_understanding_node(state: AgentState) -> dict:
    query = (state.get("query") or "").strip()
    if not query:
        return {"status": "error", "error": "Please enter a question.", "trace": _trace(state, "query_understanding")}
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]
    rewritten, warnings = query, list(state.get("warnings") or [])
    if state.get("history"):
        try:
            rewritten = rewrite_query(query, state["history"])
        except LLMConfigurationError as exc:
            return {"status": "error", "error": exc.user_message, "trace": _trace(state, "query_understanding")}
        except LLMServiceError as exc:
            warnings.append(f"Query rewrite skipped: {exc.user_message}")
            logger.warning("Query rewrite failed, using original query")
    return {"query": query, "rewritten_query": rewritten, "status": "running", "steps": 0, "retrieval_done": False,
            "retrieved_chunks": [], "tool_calls": [], "tool_outputs": [], "warnings": warnings,
            "trace": _trace(state, "query_understanding")}


# ───────────────────────── 2. intent classification ─────────────────────────
def intent_classification_node(state: AgentState) -> dict:
    trace = _trace(state, "intent_classification")
    warnings = list(state.get("warnings") or [])
    intent, reason = None, ""
    try:
        intent, reason = classify_intent(state["rewritten_query"], library_overview())
    except LLMConfigurationError as exc:
        return {"status": "error", "error": exc.user_message, "trace": trace}
    except LLMServiceError as exc:
        warnings.append(f"Intent classification fell back to default: {exc.user_message}")
    if intent is None:
        intent, reason = "document_qa", "fallback: classifier output unusable"
        warnings.append("Intent could not be classified; defaulting to document_qa.")
    return {"intent": intent, "intent_reason": reason, "warnings": warnings, "trace": trace}


# ───────────────────────── 3. agent decision (LLM chooses tools) ─────────────────────────
def agent_decision_node(state: AgentState) -> dict:
    trace = _trace(state, "agent_decision")
    warnings = list(state.get("warnings") or [])
    system = SystemMessage(content=AGENT_SYSTEM_PROMPT.format(library=library_overview(), intent=state.get("intent", "unknown")))
    msgs = [system, *history_to_messages(state.get("history")), HumanMessage(content=state["rewritten_query"]),
            *(state.get("messages") or [])]
    llm = get_llm(temperature=0).bind_tools(ALL_TOOLS)
    ai = None
    for attempt in (1, 2):
        try:
            ai = invoke_llm(llm, msgs)
            break
        except LLMToolCallError:
            logger.warning("Model produced an invalid tool call (attempt %d)", attempt)
            if attempt == 2:
                warnings.append("The model produced an invalid tool call; continuing without it.")
                ai = AIMessage(content="READY")
        except LLMConfigurationError as exc:
            return {"status": "error", "error": exc.user_message, "trace": trace}
        except LLMServiceError as exc:
            return {"status": "error", "error": exc.user_message, "trace": trace}
    calls = ai.tool_calls or []
    action = "general"
    if calls:
        action = "rag" if {c["name"] for c in calls} == {"document_search"} else "tools"
    return {"messages": [ai], "selected_action": action, "warnings": warnings, "trace": trace}


# ───────────────────────── tool execution (shared by RAG + tools nodes) ─────────────────────────
def _parse_status(content: str) -> tuple[str, str | None]:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("status") == "error":
            return "error", str(data.get("error", ""))[:500]
    except (json.JSONDecodeError, TypeError):
        pass
    return "success", None


def _merge_chunks(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {(c["doc_id"], c["chunk_index"]) for c in existing}
    merged = list(existing)
    for c in new:
        key = (c["doc_id"], c["chunk_index"])
        if key not in seen:
            seen.add(key)
            merged.append(c)
    merged.sort(key=lambda c: -c.get("score", 0))
    return merged[:MAX_CHUNKS_KEPT]


def execute_tool_calls(state: AgentState, tool_calls: list[dict]) -> dict:
    s = get_settings()
    prior = state.get("tool_calls") or []
    seen_signatures = {json.dumps([c["name"], c["args"]], sort_keys=True, default=str) for c in prior}
    tool_messages: list[ToolMessage] = []
    calls_log, outputs_log = list(prior), list(state.get("tool_outputs") or [])
    chunks = list(state.get("retrieved_chunks") or [])
    for i, tc in enumerate(tool_calls):
        name, args, call_id = tc["name"], tc.get("args") or {}, tc.get("id") or uuid.uuid4().hex
        sig = json.dumps([name, args], sort_keys=True, default=str)
        if i >= s.max_tool_calls_per_step:
            tool_messages.append(ToolMessage(content='{"status":"error","error":"Skipped: too many tool calls in one step."}',
                                             tool_call_id=call_id, name=name))
            continue
        if sig in seen_signatures:
            tool_messages.append(ToolMessage(content='{"status":"error","error":"Duplicate call: this exact call was already executed; reuse its result."}',
                                             tool_call_id=call_id, name=name))
            continue
        seen_signatures.add(sig)
        tool = TOOLS_BY_NAME.get(name)
        start = time.perf_counter()
        artifact = None
        if tool is None:
            content, status, err = json.dumps({"status": "error", "error": f"Unknown tool '{name}'."}), "error", f"Unknown tool '{name}'."
        else:
            try:
                result = tool.invoke({"name": name, "args": args, "id": call_id, "type": "tool_call"})
                content = result.content if isinstance(result.content, str) else json.dumps(result.content, default=str)
                artifact = getattr(result, "artifact", None)
                status, err = _parse_status(content)
            except Exception as exc:  # noqa: BLE001 - e.g. argument validation errors from the LLM
                logger.warning("Tool %s raised %s", name, type(exc).__name__)
                content = json.dumps({"status": "error", "error": "Invalid arguments for this tool. Check the argument names and values."})
                status, err = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
        duration = int((time.perf_counter() - start) * 1000)
        try:
            repo.log_tool_execution(tool_name=name, input_args=args, output=content, status=status, error_message=err,
                                    duration_ms=duration, conversation_id=state.get("conversation_id"))
        except SmartAssistError:
            logger.warning("Could not log tool execution for %s", name)
        if artifact:
            chunks = _merge_chunks(chunks, artifact)
        calls_log.append({"name": name, "args": args, "id": call_id})
        outputs_log.append({"tool": name, "args": args, "status": status, "error": err, "duration_ms": duration,
                            "output": truncate(content, 6000)})
        tool_messages.append(ToolMessage(content=truncate(content, 6000), tool_call_id=call_id, name=name))
    return {"messages": tool_messages, "tool_calls": calls_log, "tool_outputs": outputs_log, "retrieved_chunks": chunks,
            "steps": state.get("steps", 0) + 1}


def _last_ai_calls(state: AgentState) -> list[dict]:
    msgs = state.get("messages") or []
    if msgs and isinstance(msgs[-1], AIMessage):
        return list(msgs[-1].tool_calls or [])
    return []


# ───────────────────────── 4. RAG retrieval ─────────────────────────
def rag_retrieval_node(state: AgentState) -> dict:
    trace = _trace(state, "rag_retrieval")
    calls = _last_ai_calls(state)
    if calls:
        update = execute_tool_calls(state, calls)
        update.update(retrieval_done=True, trace=trace)
        return update
    # deterministic fallback: the agent made no tool call for a document question → retrieve anyway
    call = {"name": "document_search", "args": {"query": state["rewritten_query"], "top_k": get_settings().top_k},
            "id": f"fallback-{uuid.uuid4().hex[:8]}"}
    update = execute_tool_calls(state, [call])
    update["messages"] = [AIMessage(content="", tool_calls=[{**call, "type": "tool_call"}]), *update["messages"]]
    update.update(retrieval_done=True, selected_action="rag_fallback", trace=trace)
    return update


# ───────────────────────── 5. tool execution ─────────────────────────
def tool_execution_node(state: AgentState) -> dict:
    trace = _trace(state, "tool_execution")
    update = execute_tool_calls(state, _last_ai_calls(state))
    if any(c["name"] == "document_search" for c in _last_ai_calls(state)):
        update["retrieval_done"] = True
    update["trace"] = trace
    return update


# ───────────────────────── 6. response generation ─────────────────────────
def _fit_chunks(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[RetrievedChunk]:
    out, used = [], 0
    for c in chunks:
        rc = RetrievedChunk(**c)
        used += len(rc.content) + 80
        if used > max_chars and out:
            break
        out.append(rc)
    return out


def _format_tool_results(outputs: list[dict]) -> str:
    blocks = []
    for o in outputs:
        if o["tool"] == "document_search":
            blocks.append(f"[document_search] status={o['status']} (passages are in DOCUMENT CONTEXT)")
            continue
        blocks.append(f"[{o['tool']}] args={json.dumps(o['args'], default=str)} status={o['status']}\n{truncate(o['output'], 3500)}")
    return "\n\n".join(blocks)


def response_generation_node(state: AgentState) -> dict:
    trace = _trace(state, "response_generation")
    outputs = state.get("tool_outputs") or []
    used = _fit_chunks(state.get("retrieved_chunks") or [])
    sources = [{"ref": i, "doc_id": c.doc_id, "filename": c.filename, "page": c.page or None, "sheet": c.sheet or None,
                "score": c.score, "snippet": truncate(c.content.strip(), 320)} for i, c in enumerate(used, 1)]
    intent = state.get("intent", "general_chat")
    other_tools = any(o["tool"] != "document_search" for o in outputs)

    # Grounding guard: a document question with no relevant passages is never answered from model memory.
    if intent == "document_qa" and not used and not other_tools:
        processed = [d for d in repo.list_documents(status="processed")]
        if not processed:
            msg = ("You don't have any processed documents yet. Upload a file in **Document Library**, wait until it shows "
                   "*processed*, and ask again.")
        else:
            msg = ("I couldn't find information about that in your uploaded documents, so I won't guess. "
                   "Try rephrasing the question, naming the document, or check that the relevant file has been uploaded and processed.")
        return {"final_response": msg, "sources": [], "trace": trace}

    try:
        answer = generate_answer(query=state["rewritten_query"], intent=intent, context=format_context(used, MAX_CONTEXT_CHARS),
                                 tool_results=_format_tool_results(outputs), history=state.get("history") or [])
    except SmartAssistError as exc:
        return {"status": "error", "error": exc.user_message, "sources": sources, "trace": trace}
    return {"final_response": answer or "I could not produce an answer.", "sources": sources, "trace": trace}


# ───────────────────────── 7. final response ─────────────────────────
def final_response_node(state: AgentState) -> dict:
    trace = _trace(state, "final_response")
    names = {c["name"] for c in (state.get("tool_calls") or [])}
    if not names:
        action = "general"
    elif names == {"document_search"}:
        action = "rag"
    elif "document_search" in names:
        action = "rag+tools"
    else:
        action = "tools"
    if state.get("status") == "error":
        err = state.get("error") or "Something went wrong while processing your request."
        return {"final_response": f"⚠️ {err}", "status": "error", "selected_action": action, "trace": trace}
    status = "completed"
    answer = state.get("final_response") or ""
    msgs = state.get("messages") or []
    unfinished = bool(msgs) and (isinstance(msgs[-1], ToolMessage) or bool(_last_ai_calls(state)))
    if unfinished and state.get("steps", 0) >= get_settings().max_agent_steps:
        status = "step_limit"
        answer += f"\n\n_Note: I stopped after {get_settings().max_agent_steps} tool-calling steps to avoid looping; the answer may be incomplete._"
    return {"final_response": answer, "status": status, "selected_action": action, "trace": trace}
