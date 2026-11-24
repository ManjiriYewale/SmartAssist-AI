"""Chains used by the agent nodes: query rewrite, intent classification, final answer."""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from backend.chains.prompts import INTENTS, INTENT_PROMPT, QUERY_REWRITE_PROMPT, RESPONSE_PROMPT
from backend.config.settings import get_settings
from backend.llm.model import get_llm, invoke_llm


def history_to_messages(history: list[dict] | None, limit: int | None = None) -> list[BaseMessage]:
    limit = get_settings().max_history_messages if limit is None else limit
    out: list[BaseMessage] = []
    for m in (history or [])[-limit:] if limit else []:
        text = (m.get("content") or "")[:1500]
        out.append(HumanMessage(content=text) if m.get("role") == "user" else AIMessage(content=text))
    return out


def rewrite_query(query: str, history: list[dict]) -> str:
    chain = QUERY_REWRITE_PROMPT | get_llm(temperature=0) | StrOutputParser()
    text = invoke_llm(chain, {"query": query, "history": history_to_messages(history, 6)}).strip().strip('"\'`')
    if not text or len(text) > max(400, 3 * len(query)):
        return query
    return text


def parse_intent(raw: str) -> tuple[str | None, str]:
    """Extract the intent from the model output (JSON preferred, tolerant of fences / prose)."""
    text = raw or ""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            intent = str(data.get("intent", "")).strip().lower()
            if intent in INTENTS:
                return intent, str(data.get("reason", ""))[:200]
        except (json.JSONDecodeError, AttributeError):
            pass
    low = text.lower()
    for name in INTENTS:
        if name in low:
            return name, ""
    return None, ""


def classify_intent(query: str, library: str) -> tuple[str | None, str]:
    chain = INTENT_PROMPT | get_llm(temperature=0)
    msg = invoke_llm(chain, {"query": query, "library": library})
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return parse_intent(content)


def generate_answer(*, query: str, intent: str, context: str, tool_results: str, history: list[dict]) -> str:
    chain = RESPONSE_PROMPT | get_llm(temperature=0.2) | StrOutputParser()
    return invoke_llm(chain, {
        "query": query, "intent": intent, "history": history_to_messages(history),
        "context": context or "(none)", "tool_results": tool_results or "(none)",
    }).strip()
