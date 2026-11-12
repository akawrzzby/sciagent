"""Database models.

The schema is intentionally relational and explicit: every artefact that the
parser produces (sections, tables, figures, formulas, references) is stored with
its **page number and bounding box** so that any generated answer can be traced
back to an exact location in the original PDF.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# --------------------------------------------------------------------------- papers
class Paper(Base):
    """A parsed academic paper (the canonical library entry)."""

    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512), default="")
    authors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    affiliations: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str] = mapped_column(String(256), default="")
    doi: Mapped[str] = mapped_column(String(128), default="")
    arxiv_id: Mapped[str] = mapped_column(String(64), default="", index=True)

    # --- file / content identity -------------------------------------------
    filename: Mapped[str] = mapped_column(String(512), default="")
    source_path: Mapped[str] = mapped_column(String(1024), default="")
    file_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)

    # --- parsing state ------------------------------------------------------
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    parse_error: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    is_scanned: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_pages: Mapped[int] = mapped_column(Integer, default=0)
    # Detected version label ("v1", "v7", ...); empty when not an arXiv paper.
    version_label: Mapped[str] = mapped_column(String(32), default="", index=True)
    # Similarity against the closest existing paper (0..1) at upload time.
    dedup_score: Mapped[float] = mapped_column(Float, default=0.0)
    dedup_note: Mapped[str] = mapped_column(Text, default="")

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                onupdate=utcnow)

    sections: Mapped[list["Section"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Section.order_index"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Chunk.ordinal"
    )
    references: Mapped[list["PaperReference"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    # The tables below are only reachable through ``paper_id``.  They still need
    # an ORM relationship: the unit of work derives the INSERT order from the
    # relationship graph, *not* from the raw ForeignKey constraints.  Without
    # these, a child row can be emitted before the parent row exists and
    # SQLite rejects the whole flush with "FOREIGN KEY constraint failed".
    blocks: Mapped[list["ContentBlock"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="ContentBlock.order_index"
    )
    tables: Mapped[list["PaperTable"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    figures: Mapped[list["PaperFigure"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    formulas: Mapped[list["PaperFormula"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    versions: Mapped[list["PaperVersion"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    translations: Mapped[list["Translation"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_papers_arxiv_version", "arxiv_id", "version_label"),)


class PaperVersion(Base):
    """Historical versions of the same paper kept side by side for diffing."""

    __tablename__ = "paper_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    version_label: Mapped[str] = mapped_column(String(32), default="")
    filename: Mapped[str] = mapped_column(String(512), default="")
    file_hash: Mapped[str] = mapped_column(String(64), default="")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    # LLM generated summary of what changed between this and the other version.
    diff_summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    paper: Mapped[Paper] = relationship(back_populates="versions")


class Section(Base):
    """A node of the heading tree (level 1/2/3)."""

    __tablename__ = "sections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    # Self reference for the heading tree (level 1 -> level 2 -> level 3).
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("sections.id", ondelete="SET NULL"), nullable=True, index=True
    )
    level: Mapped[int] = mapped_column(Integer, default=1)
    number: Mapped[str] = mapped_column(String(32), default="")      # e.g. "3.2.2"
    title: Mapped[str] = mapped_column(String(512), default="")
    page_start: Mapped[int] = mapped_column(Integer, default=0)
    page_end: Mapped[int] = mapped_column(Integer, default=0)
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    content: Mapped[str] = mapped_column(Text, default="")
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    paper: Mapped[Paper] = relationship(back_populates="sections")
    parent: Mapped["Section | None"] = relationship(remote_side="Section.id",
                                                    back_populates="children")
    children: Mapped[list["Section"]] = relationship(back_populates="parent")


class ContentBlock(Base):
    """A positioned content block (text paragraph, table, figure, formula).

    ``bbox`` uses PDF points with the origin at the top-left of the page.
    """

    __tablename__ = "content_blocks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    page_no: Mapped[int] = mapped_column(Integer, default=0, index=True)
    block_type: Mapped[str] = mapped_column(String(24), default="text", index=True)
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    content: Mapped[str] = mapped_column(Text, default="")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_blocks_paper_page", "paper_id", "page_no"),)

    paper: Mapped[Paper] = relationship(back_populates="blocks")


class PaperTable(Base):
    """An extracted table."""

    __tablename__ = "paper_tables"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    caption: Mapped[str] = mapped_column(Text, default="")
    markdown: Mapped[str] = mapped_column(Text, default="")
    csv_text: Mapped[str] = mapped_column(Text, default="")
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)

    paper: Mapped[Paper] = relationship(back_populates="tables")


class PaperFigure(Base):
    """An extracted figure / chart image."""

    __tablename__ = "paper_figures"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    caption: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str] = mapped_column(String(1024), default="")
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)
    # VLM-generated description of the figure content (multimodal understanding).
    description: Mapped[str] = mapped_column(Text, default="")

    paper: Mapped[Paper] = relationship(back_populates="figures")


class PaperFormula(Base):
    """A detected formula, optionally transcribed to LaTeX by the VLM."""

    __tablename__ = "paper_formulas"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    latex: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    bbox: Mapped[list[float]] = mapped_column(JSON, default=list)

    paper: Mapped[Paper] = relationship(back_populates="formulas")


class PaperReference(Base):
    """A parsed bibliography entry."""

    __tablename__ = "paper_references"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    index_in_doc: Mapped[int] = mapped_column(Integer, default=0)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    title: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str] = mapped_column(String(512), default="")
    doi: Mapped[str] = mapped_column(String(128), default="")
    arxiv_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    # Set when the reference could be aligned with a paper in the local library.
    linked_paper_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)

    paper: Mapped[Paper] = relationship(back_populates="references")


class Chunk(Base):
    """A retrievable text chunk with its provenance."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    # Path of the memmap holding the dense vector for this chunk.
    embedding_ref: Mapped[str] = mapped_column(String(512), default="")
    embedding_row: Mapped[int] = mapped_column(Integer, default=-1)
    kind: Mapped[str] = mapped_column(String(32), default="body")

    paper: Mapped[Paper] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_chunk_paper_ordinal"),
        Index("ix_chunks_paper_ordinal", "paper_id", "ordinal"),
    )


class CitationEdge(Base):
    """A directed citation edge between two library papers."""

    __tablename__ = "citation_edges"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    src_paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    dst_paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    reference_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # 1 when the citing context (surrounding sentence) is topically close to the
    # cited work; used to filter out "perfunctory" citations.
    relevance: Mapped[float] = mapped_column(Float, default=0.0)
    context: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("src_paper_id", "dst_paper_id", name="uq_citation_edge"),)


class Artifact(Base):
    """A generated artefact (TikZ / Mermaid / Graphviz / figure / caption)."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # tikz|mermaid|graphviz|figure|caption|framework
    title: Mapped[str] = mapped_column(String(512), default="")
    payload: Mapped[str] = mapped_column(Text, default="")     # source code / markdown
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    """A long running background job (parsing, indexing, review generation)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    target_id: Mapped[str] = mapped_column(String(32), default="", index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                onupdate=utcnow)


class Translation(Base):
    """A cached full-document translation (Markdown, code/formulas kept in EN)."""

    __tablename__ = "translations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    locale: Mapped[str] = mapped_column(String(16), default="zh")
    markdown: Mapped[str] = mapped_column(Text, default="")
    segments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("paper_id", "locale", name="uq_translation"),)

    paper: Mapped[Paper] = relationship(back_populates="translations")


class ChatMessage(Base):
    """One turn of an agent conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RecommendationFeedback(Base):
    """Thumb up / down on a recommended paper (feeds the recommender)."""

    __tablename__ = "recommendation_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[int] = mapped_column(Integer, default=0)  # +1 / -1
    title: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "arxiv_id", name="uq_feedback_user_arxiv"),)


class AuditRecord(Base):
    """Prompt audit trail (one row per LLM call).

    The competition's final round asks for a "Prompt audit record", therefore we
    persist the full call: system prompt, user prompt, tools, raw response,
    token usage and latency.
    """

    __tablename__ = "audit_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    purpose: Mapped[str] = mapped_column(String(64), default="", index=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt: Mapped[str] = mapped_column(Text, default="")
    tools: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    response: Mapped[str] = mapped_column(Text, default="")
    reasoning: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
