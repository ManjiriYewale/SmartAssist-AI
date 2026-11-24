"""LangGraph StateGraph assembly and the public ``run_agent`` entry point."""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from backend.agents import nodes
from backend.agents.routing import route_after_agent, route_after_intent, route_after_tools
from backend.agents.state import AgentState
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("query_understanding", nodes.query_understanding_node)
    g.add_node("intent_classification", nodes.intent_classification_node)
    g.add_node("agent_decision", nodes.agent_decision_node)
    g.add_node("rag_retrieval", nodes.rag_retrieval_node)
    g.add_node("tool_execution", nodes.tool_execution_node)
    g.add_node("response_generation", nodes.response_generation_node)
    g.add_node("final_response", nodes.final_response_node)

    g.add_edge(START, "query_understanding")
    g.add_conditional_edges("query_understanding", lambda s: "error" if s.get("status") == "error" else "ok",
                            {"ok": "intent_classification", "error": "final_response"})
    g.add_conditional_edges("intent_classification", route_after_intent,
                            {"general": "response_generation", "agent": "agent_decision", "error": "final_response"})
    g.add_conditional_edges("agent_decision", route_after_agent,
                            {"rag": "rag_retrieval", "tools": "tool_execution",
                             "respond": "response_generation", "error": "final_response"})
    g.add_conditional_edges("rag_retrieval", route_after_tools, {"agent": "agent_decision", "respond": "response_generation"})
    g.add_conditional_edges("tool_execution", route_after_tools, {"agent": "agent_decision", "respond": "response_generation"})
    g.add_edge("response_generation", "final_response")
    g.add_edge("final_response", END)
    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph().compile()


def run_agent(query: str, *, history: list[dict] | None = None, conversation_id: int | None = None) -> AgentState:
    """Execute the workflow for one user message and return the final state."""
    graph = get_compiled_graph()
    initial: AgentState = {"query": query, "history": history or [], "conversation_id": conversation_id,
                           "messages": [], "trace": [], "warnings": [], "status": "running"}
    try:
        # each tool round = 2 nodes (agent + tool); +8 covers the fixed nodes. Hard stop against runaway loops.
        limit = 2 * get_settings().max_agent_steps + 12
        return graph.invoke(initial, config={"recursion_limit": limit})
    except Exception as exc:  # noqa: BLE001 - includes GraphRecursionError
        logger.exception("Agent graph failed")
        msg = "The assistant took too many steps and was stopped." if type(exc).__name__ == "GraphRecursionError" \
            else "The assistant hit an unexpected error. Please try again."
        return {**initial, "status": "error", "error": msg, "final_response": f"⚠️ {msg}", "sources": [],
                "tool_calls": [], "tool_outputs": [], "retrieved_chunks": [], "intent": "unknown", "selected_action": "general"}
