"""HTTP surface of the re-laid-out document (the reading view).

The reader renders a paper as a continuous document instead of as a stack of
page images, so the frontend needs the whole paper in one response.  Building
it is a pure database read (the parse and the translation are already
persisted), which is why there is no job endpoint here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.render.document import build_document

router = APIRouter(prefix="/reader", tags=["reader"])


@router.get("/{paper_id}/document")
async def get_document(paper_id: str, lang: str = "zh") -> dict[str, Any]:
    """The paper as a flat, renderable document.

    ``lang`` selects the translation to attach; a language the paper was never
    translated into yields the document with empty targets.
    """
    document = await build_document(paper_id, lang)
    if document is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return document.to_dict()
