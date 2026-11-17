"""The model endpoints the app can talk to, and which one is active.

A deployment usually has one model, but a demo does not: the reader shows the
name of the model that is answering, and the operator may want to point it at
a hosted API instead of the local server.  The profiles live in
``<data_dir>/models.json``; the file is created from the environment settings
the first time the app runs, so a fresh install behaves exactly as before.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Profile:
    """One OpenAI-compatible endpoint."""

    id: str
    label: str
    base_url: str
    model: str
    api_key: str = ""

    def to_public(self) -> dict[str, Any]:
        """The profile as the UI may see it: the key is never sent back."""
        data = asdict(self)
        key = data.pop("api_key")
        data["has_key"] = bool(key)
        data["key_hint"] = f"…{key[-4:]}" if len(key) >= 4 else ""
        return data


def _file() -> Path:
    return Path(get_settings().data_dir) / "models.json"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _seed() -> dict[str, Any]:
    settings = get_settings()
    profile = Profile(
        id="default",
        label=settings.llm_model or "本地模型",
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )
    return {"active": profile.id, "profiles": [asdict(profile)]}


def load() -> dict[str, Any]:
    """Read the profiles; a missing or broken file falls back to the settings."""
    path = _file()
    if not path.exists():
        data = _seed()
        save(data)
        return data
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("models.json is unreadable (%s); falling back to settings", exc)
        return _seed()
    if not isinstance(data, dict) or not data.get("profiles"):
        return _seed()
    return data


def save(data: dict[str, Any]) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def profiles() -> list[Profile]:
    return [Profile(**item) for item in load().get("profiles", [])]


def active() -> Profile | None:
    """The profile the client should use, or ``None`` for the environment one."""
    data = load()
    wanted = data.get("active")
    for profile in profiles():
        if profile.id == wanted:
            return profile
    return profiles()[0] if profiles() else None


def add(label: str, base_url: str, model: str, api_key: str = "") -> Profile:
    profile = Profile(id=_new_id(), label=label.strip() or model.strip(),
                      base_url=base_url.strip().rstrip("/"), model=model.strip(),
                      api_key=api_key.strip())
    data = load()
    data.setdefault("profiles", []).append(asdict(profile))
    save(data)
    return profile


def update(profile_id: str, **fields: Any) -> Profile | None:
    data = load()
    for item in data.get("profiles", []):
        if item.get("id") != profile_id:
            continue
        for key in ("label", "base_url", "model", "api_key"):
            if fields.get(key) is not None:
                item[key] = str(fields[key]).strip()
        item["base_url"] = item.get("base_url", "").rstrip("/")
        save(data)
        return Profile(**item)
    return None


def activate(profile_id: str) -> bool:
    data = load()
    if not any(item.get("id") == profile_id for item in data.get("profiles", [])):
        return False
    data["active"] = profile_id
    save(data)
    return True


def remove(profile_id: str) -> bool:
    data = load()
    kept = [item for item in data.get("profiles", []) if item.get("id") != profile_id]
    if len(kept) == len(data.get("profiles", [])):
        return False
    data["profiles"] = kept
    if data.get("active") == profile_id and kept:
        data["active"] = kept[0]["id"]
    save(data)
    return True
