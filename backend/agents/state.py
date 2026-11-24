"""Shared LangGraph agent state."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # input
    query: str
    history: list[dict]                 # prior turns: [{"role": "user"|"assistant", "content": str}]
    conversation_id: Optional[int]
    # understanding / routing
    rewritten_query: str
    intent: str
    intent_reason: str
    selected_action: str                # general | rag | tools | rag+tools
    # agent loop
    messages: Annotated[list[BaseMessage], add_messages]   # AI tool-call messages + ToolMessages
    steps: int                          # tool-execution rounds performed
    retrieval_done: bool
    # results
    retrieved_chunks: list[dict]
    tool_calls: list[dict]
    tool_outputs: list[dict]
    final_response: str
    sources: list[dict]
    # control
    status: str                         # running | completed | step_limit | error
    error: Optional[str]
    warnings: list[str]
    trace: list[str]                    # node visit order (for debugging / tests)
