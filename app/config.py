"""Runtime configuration, all from environment variables."""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


DATA_DIR = Path(_env("COUNCIL_DATA_DIR", "./data")).resolve()
# Keep the Hugging Face model cache next to the voices so one volume holds
# everything (RunPod mounts its volume at /workspace, Docker Compose at /data).
os.environ.setdefault("HF_HOME", str(DATA_DIR / "huggingface"))
REFS_DIR = DATA_DIR / "refs"          # original reference clips
PROMPTS_DIR = DATA_DIR / "prompts"    # pre-encoded VoiceClonePrompt .pt files
REGISTRY_FILE = DATA_DIR / "voices.json"

MODEL_ID = _env("COUNCIL_MODEL", "k2-fsa/OmniVoice")
DEVICE = _env("COUNCIL_DEVICE", "auto")          # auto | cuda | mps | cpu
PRELOAD = _env("COUNCIL_PRELOAD", "1") == "1"      # load the model at startup
API_KEY = os.environ.get("COUNCIL_API_KEY", "").strip()  # empty = no auth

DEFAULT_LANGUAGE = _env("COUNCIL_DEFAULT_LANGUAGE", "pt")
DEFAULT_FORMAT = _env("COUNCIL_DEFAULT_FORMAT", "mp3")
DEFAULT_NUM_STEP = int(_env("COUNCIL_NUM_STEP", "16"))
DEFAULT_GUIDANCE = float(_env("COUNCIL_GUIDANCE_SCALE", "2.0"))
MAX_TEXT_CHARS = int(_env("COUNCIL_MAX_TEXT_CHARS", "4096"))
MAX_REF_SECONDS = float(_env("COUNCIL_MAX_REF_SECONDS", "20"))
NORMALIZE_DBFS = float(_env("COUNCIL_NORMALIZE_DBFS", "-2.0"))

# Denoise/dereverb reference clips with DeepFilterNet before encoding them.
# Zero-shot cloning copies the room acoustics of the reference, so a podcast
# or stage clip clones with its reverb unless it is cleaned first.
CLEAN_REFS = _env("COUNCIL_CLEAN_REFS", "1") == "1"
# Interpreter that has `deepfilternet` installed (defaults to this one).
DFN_PYTHON = _env("COUNCIL_DFN_PYTHON", "")

# Keep the model resident forever by default (the council site hits it
# intermittently and a reload costs ~10 s + VRAM churn).
IDLE_UNLOAD_SECONDS = int(_env("COUNCIL_IDLE_UNLOAD_SECONDS", "0"))
