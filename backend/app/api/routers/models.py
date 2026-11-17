"""The model endpoints: list them, add one, make one active, remove one.

The browser never receives an API key back; it only learns whether a key is
set and its last four characters.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.llm import providers

router = APIRouter(prefix="/models", tags=["models"])


class ProfileIn(BaseModel):
    label: str = ""
    base_url: str = Field(min_length=4)
    model: str = Field(min_length=1)
    api_key: str = ""


class ProbeIn(BaseModel):
    base_url: str
    model: str = ""
    api_key: str = ""


@router.get("")
async def list_models() -> dict[str, Any]:
    data = providers.load()
    return {
        "active": data.get("active"),
        "profiles": [profile.to_public() for profile in providers.profiles()],
    }


@router.post("")
async def create_model(body: ProfileIn) -> dict[str, Any]:
    profile = providers.add(body.label, body.base_url, body.model, body.api_key)
    return profile.to_public()


@router.post("/{profile_id}/activate")
async def activate_model(profile_id: str) -> dict[str, Any]:
    if not providers.activate(profile_id):
        raise HTTPException(status_code=404, detail="profile not found")
    return {"active": profile_id}


@router.patch("/{profile_id}")
async def patch_model(profile_id: str, body: dict[str, Any]) -> dict[str, Any]:
    profile = providers.update(profile_id, **body)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return profile.to_public()


@router.delete("/{profile_id}")
async def delete_model(profile_id: str) -> dict[str, Any]:
    if not providers.remove(profile_id):
        raise HTTPException(status_code=404, detail="profile not found")
    return {"deleted": profile_id}


@router.post("/probe")
async def probe_model(body: ProbeIn) -> dict[str, Any]:
    """Check an endpoint before saving it: does it answer, and with which models?"""
    url = body.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {body.api_key}"} if body.api_key else {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=headers)
        if response.status_code >= 400:
            return {"ok": False, "error": f"HTTP {response.status_code}"}
        payload = response.json()
        ids = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        return {"ok": True, "models": ids[:40]}
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        return {"ok": False, "error": str(exc)[:200]}
