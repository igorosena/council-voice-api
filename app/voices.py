"""Voice registry: a JSON file plus the reference clip and the cached prompt."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import config

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_lock = threading.Lock()


@dataclass
class Voice:
    id: str
    name: str
    language: str
    ref_text: str
    ref_audio: str      # path relative to DATA_DIR
    prompt: str         # path relative to DATA_DIR
    ref_seconds: float
    cleaned: bool = False
    clean_error: Optional[str] = None

    @property
    def ref_audio_path(self) -> Path:
        return config.DATA_DIR / self.ref_audio

    @property
    def prompt_path(self) -> Path:
        return config.DATA_DIR / self.prompt

    def public(self) -> dict:
        d = asdict(self)
        d.pop("ref_audio")
        d.pop("prompt")
        return d


def validate_id(voice_id: str) -> str:
    voice_id = voice_id.strip().lower()
    if not _SLUG_RE.match(voice_id):
        raise ValueError(
            "voice id must be 1-40 chars of a-z, 0-9, '-' or '_' (e.g. 'igor')"
        )
    return voice_id


def _read() -> dict[str, Voice]:
    if not config.REGISTRY_FILE.exists():
        return {}
    raw = json.loads(config.REGISTRY_FILE.read_text("utf-8"))
    return {k: Voice(**v) for k, v in raw.items()}


def _write(voices: dict[str, Voice]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.REGISTRY_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({k: asdict(v) for k, v in voices.items()}, indent=2, ensure_ascii=False),
        "utf-8",
    )
    tmp.replace(config.REGISTRY_FILE)


def list_voices() -> list[Voice]:
    with _lock:
        return sorted(_read().values(), key=lambda v: v.id)


def get_voice(voice_id: str) -> Optional[Voice]:
    with _lock:
        return _read().get(voice_id.strip().lower())


def upsert(voice: Voice) -> None:
    with _lock:
        voices = _read()
        voices[voice.id] = voice
        _write(voices)


def delete(voice_id: str) -> Optional[Voice]:
    with _lock:
        voices = _read()
        voice = voices.pop(voice_id, None)
        if voice is None:
            return None
        _write(voices)
    for p in (voice.ref_audio_path, voice.prompt_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return voice
