"""Conditional-edge functions of the LangGraph workflow (pure functions of the state)."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.agents.state import AgentState
from backend.config.settings import get_settings

RETRIEVAL_INTENTS = {"document_qa"}


def _pending_calls(state: AgentState) -> list[dict]:
    msgs = state.get("messages") or []
    if msgs and isinstance(msgs[-1], AIMessage):
        return list(msgs[-1].tool_calls or [])
    return []


def route_after_intent(state: AgentState) -> str:
    if state.get("status") == "error":
        return "error"
    return "general" if state.get("intent") == "general_chat" else "agent"


def route_after_agent(state: AgentState) -> str:
    if state.get("status") == "error":
        return "error"
    calls = _pending_calls(state)
    if calls:
        if state.get("steps", 0) >= get_settings().max_agent_steps:
            return "respond"          # execution limit reached – never loop forever
        names = {c["name"] for c in calls}
        return "rag" if names == {"document_search"} else "tools"
    # the agent chose no tool. Document questions must still be grounded → deterministic retrieval fallback
    if state.get("intent") in RETRIEVAL_INTENTS and not state.get("retrieval_done"):
        return "rag"
    return "respond"


def route_after_tools(state: AgentState) -> str:
    if state.get("status") == "error":
        return "respond"
    if state.get("selected_action") == "rag_fallback":
        return "respond"
    if state.get("steps", 0) >= get_settings().max_agent_steps:
        return "respond"
    return "agent"
