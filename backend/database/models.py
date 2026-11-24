"""SQLAlchemy ORM models (PostgreSQL in production, any SQLAlchemy DB in tests)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="uploaded", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    extra_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    tool_name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    input_args: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="agent")  # agent | manual
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(40), nullable=True)
    selected_action: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tools_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    num_sources: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
