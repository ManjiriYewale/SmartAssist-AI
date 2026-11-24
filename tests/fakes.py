"""A scripted chat model so the agent, tools and API can be tested offline.

It implements the real LangChain ``BaseChatModel`` interface (including ``bind_tools``),
so the graph runs exactly as it would with ChatGroq – only the model's decisions are scripted.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChat(BaseChatModel):
    intent: str = "document_qa"
    plan: list = []                  # list of steps; each step = list of (tool_name, args) tuples
    answer: str = "FAKE ANSWER"
    fail_on: str = ""                # "classify" | "agent" | "response" → raise a rate-limit style error
    always_call: Any = None          # (tool_name, args) returned on every agent call (loop test)
    bound: bool = False
    log: list = []                   # shared call log: (role, text)
    counter: dict = {}

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound": True})

    def _role(self, messages: list[BaseMessage]) -> str:
        sys = str(messages[0].content) if messages else ""
        if self.bound:
            return "agent"
        if "router of a document assistant" in sys:
            return "classify"
        if "rewrite the user's latest message" in sys:
            return "rewrite"
        if "SmartAssist AI, a precise" in sys:
            return "response"
        if "condense one section" in sys:
            return "map"
        if "comparing two documents" in sys:
            return "compare"
        if "expert document analyst" in sys:
            return "summary"
        return "other"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        role = self._role(messages)
        self.log.append((role, "\n".join(str(m.content) for m in messages)))
        if self.fail_on == role:
            raise RuntimeError("429 rate limit exceeded")
        if role == "classify":
            msg = AIMessage(content=f'{{"intent": "{self.intent}", "reason": "scripted"}}')
        elif role == "agent":
            step = None
            if self.always_call:
                self.counter["n"] = self.counter.get("n", 0) + 1
                call = self.always_call(self.counter["n"]) if callable(self.always_call) else self.always_call
                step = [call]
            elif self.plan:
                step = self.plan.pop(0)
            if step:
                msg = AIMessage(content="", tool_calls=[
                    {"name": n, "args": dict(a), "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"} for n, a in step])
            else:
                msg = AIMessage(content="READY")
        elif role == "rewrite":
            last = str(messages[-1].content).split("Latest message:")[-1].split("\n")[0].strip()
            msg = AIMessage(content=f"REWRITTEN: {last}")
        elif role == "map":
            msg = AIMessage(content="- condensed notes")
        elif role in {"summary", "compare"}:
            msg = AIMessage(content=f"[{role}] generated text")
        else:
            msg = AIMessage(content=self.answer)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_llm(**kw) -> ScriptedChat:
    kw.setdefault("log", [])
    kw.setdefault("plan", [])
    kw.setdefault("counter", {})
    return ScriptedChat(**kw)
