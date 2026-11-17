"""The document model: one paper, re-laid out for reading.

This is the contract between the parser/renderer (backend) and the reading view
(frontend).  A paper is a flat list of blocks in reading order; each block knows
where it came from (page, bbox) so the reader can still jump to the source page.

Nothing here knows about the database or about HTTP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Sentence:
    """One aligned sentence pair."""

    id: str
    source: str
    target: str = ""
    block_id: str = ""


@dataclass(slots=True)
class Cell:
    """A table cell.

    ``rowspan``/``colspan`` default to 1; a merged cell appears once and the
    cells it covers are omitted from the row.
    """

    text: str
    target: str = ""
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


@dataclass(slots=True)
class Block:
    """A block of the re-laid-out document.

    ``kind`` is one of ``heading``, ``paragraph``, ``list``, ``table``,
    ``figure``, ``formula``, ``footnote``, ``reference``.
    """

    kind: str
    page_no: int = 0
    bbox: list[float] = field(default_factory=list)
    id: str = ""
    # heading / paragraph / list / footnote / reference
    level: int = 0
    text: str = ""
    target: str = ""
    sentences: list[Sentence] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    # table
    caption: str = ""
    caption_target: str = ""
    header_rows: int = 0
    rows: list[list[Cell]] = field(default_factory=list)
    image_url: str = ""
    # figure / formula
    latex: str = ""
    number: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "page_no": self.page_no,
                                "id": self.id, "bbox": self.bbox}
        if self.kind == "heading":
            data.update(level=self.level, text=self.text, target=self.target)
        elif self.kind in ("paragraph", "footnote", "reference"):
            data.update(text=self.text, target=self.target,
                        sentences=[asdict(s) for s in self.sentences])
        elif self.kind == "list":
            data.update(items=self.items, sentences=[asdict(s) for s in self.sentences])
        elif self.kind == "table":
            data.update(
                caption=self.caption, caption_target=self.caption_target,
                header_rows=self.header_rows, image_url=self.image_url,
                rows=[[asdict(cell) for cell in row] for row in self.rows],
            )
        elif self.kind == "figure":
            data.update(caption=self.caption, caption_target=self.caption_target,
                        image_url=self.image_url)
        elif self.kind == "formula":
            data.update(latex=self.latex, image_url=self.image_url,
                        number=self.number)
        if self.note:
            data["note"] = self.note
        return data


@dataclass(slots=True)
class Document:
    """A paper, ready to render."""

    paper_id: str
    title: str
    title_target: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    abstract_target: str = ""
    lang: str = "zh"
    blocks: list[Block] = field(default_factory=list)
    #: total counts, for a table of contents and for quality checks
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "title_target": self.title_target,
            "authors": self.authors,
            "abstract": self.abstract,
            "abstract_target": self.abstract_target,
            "lang": self.lang,
            "counts": self.counts,
            "blocks": [block.to_dict() for block in self.blocks],
        }
