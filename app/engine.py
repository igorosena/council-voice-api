"""Model lifecycle, voice-prompt encoding, synthesis and audio encoding.

One resident OmniVoice model, one generation at a time (the model is not
thread-safe and a second concurrent job only slows both down).
"""
from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from . import config

logger = logging.getLogger("council.engine")

_model = None
_model_lock = threading.Lock()      # guards load / unload
_gen_lock = threading.Lock()        # one synthesis at a time
_prompt_cache: dict[str, object] = {}
_last_used = time.monotonic()

FORMATS = {
    "wav": ("audio/wav", "wav"),
    "mp3": ("audio/mpeg", "mp3"),
    "opus": ("audio/ogg", "ogg"),
    "flac": ("audio/flac", "flac"),
    "pcm": ("audio/pcm", "pcm"),
}


# ---------------------------------------------------------------- device / dtype
def resolve_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    from omnivoice.utils.dtype import tts_dtype_name

    return getattr(torch, tts_dtype_name(device))


# ---------------------------------------------------------------- model
def get_model():
    """Return the resident model, loading it on first use."""
    global _model, _last_used
    _last_used = time.monotonic()
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from omnivoice import OmniVoice

        device = resolve_device()
        t0 = time.perf_counter()
        logger.info("loading %s on %s ...", config.MODEL_ID, device)
        model = OmniVoice.from_pretrained(
            config.MODEL_ID, device_map=device, dtype=_dtype_for(device)
        )
        model.eval()
        logger.info("model ready in %.1fs (sr=%d)", time.perf_counter() - t0, model.sampling_rate)
        _model = model
        return _model


def is_loaded() -> bool:
    return _model is not None


def unload_if_idle() -> None:
    """Called by a background timer when COUNCIL_IDLE_UNLOAD_SECONDS > 0."""
    global _model
    if not config.IDLE_UNLOAD_SECONDS or _model is None:
        return
    if time.monotonic() - _last_used < config.IDLE_UNLOAD_SECONDS:
        return
    with _model_lock, _gen_lock:
        logger.info("unloading idle model")
        _model = None
        _prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- reference clips
def probe_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def convert_to_wav(src: Path, dst: Path) -> None:
    """Normalise any upload (mp3/m4a/ogg/webm/wav) to 24 kHz mono 16-bit WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; it is required to ingest reference clips")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)],
        check=True,
    )


def clean_reference(wav_path: Path) -> tuple[bool, Optional[str]]:
    """Denoise + dereverb ``wav_path`` in place with DeepFilterNet.

    Returns ``(True, None)`` when the clip was cleaned, or ``(False, reason)``
    when cleaning is disabled or failed (the original clip is kept).
    """
    if not config.CLEAN_REFS:
        return False, "disabled (COUNCIL_CLEAN_REFS=0)"
    import sys

    python = config.DFN_PYTHON or sys.executable
    script = Path(__file__).resolve().parent.parent / "scripts" / "clean_ref.py"
    tmp48 = wav_path.with_name(wav_path.stem + ".dfn48.wav")
    try:
        proc = subprocess.run(
            [python, str(script), str(wav_path), str(tmp48)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            tail = [l for l in proc.stderr.strip().splitlines() if l.strip()][-3:]
            reason = " | ".join(tail) or f"exit code {proc.returncode}"
            logger.warning("DeepFilterNet failed, keeping raw reference: %s", reason)
            return False, reason[:500]
        convert_to_wav(tmp48, wav_path)
        return True, None
    except Exception as e:  # timeout, missing interpreter, ffmpeg failure
        logger.warning("DeepFilterNet failed, keeping raw reference: %s", e)
        return False, str(e)[:500]
    finally:
        tmp48.unlink(missing_ok=True)


def encode_prompt(ref_wav: Path, ref_text: str, out_path: Path) -> None:
    """Encode a reference clip into a reusable VoiceClonePrompt and save it."""
    model = get_model()
    with _gen_lock:
        prompt = model.create_voice_clone_prompt(str(ref_wav), ref_text=ref_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt.save(str(out_path))
    _prompt_cache.pop(str(out_path), None)


def _load_prompt(path: Path):
    key = str(path)
    prompt = _prompt_cache.get(key)
    if prompt is None:
        from omnivoice.models.omnivoice import VoiceClonePrompt

        prompt = VoiceClonePrompt.load(key)
        _prompt_cache[key] = prompt
    return prompt


# ---------------------------------------------------------------- synthesis
def normalize_language(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.strip()
    if lang.lower() in ("auto", "none", ""):
        return None
    # "pt-BR" / "pt_BR" -> "pt"; names like "Portuguese" pass through.
    if "-" in lang or "_" in lang:
        lang = lang.replace("_", "-").split("-")[0]
    return lang.lower() if len(lang) <= 3 else lang


def _normalize_audio(audio: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    """Peak-normalise, never amplifying a near-silent (failed) render."""
    if audio.numel() == 0:
        return audio
    peak = audio.abs().max()
    if peak > 10 ** (-50.0 / 20.0):
        audio = audio * (10 ** (target_dbfs / 20.0) / peak)
    return audio


def synthesize(
    prompt_path: Path,
    text: str,
    language: Optional[str],
    speed: float = 1.0,
    num_step: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, int, float]:
    """Return (mono waveform (T,), sample_rate, generation_seconds)."""
    model = get_model()
    prompt = _load_prompt(prompt_path)
    kwargs = dict(
        num_step=num_step or config.DEFAULT_NUM_STEP,
        guidance_scale=guidance_scale if guidance_scale is not None else config.DEFAULT_GUIDANCE,
    )
    with _gen_lock:
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        audios = model.generate(
            text=text,
            language=normalize_language(language),
            voice_clone_prompt=prompt,
            speed=None if math.isclose(speed, 1.0) else speed,
            **kwargs,
        )
        gen_s = time.perf_counter() - t0
    audio = audios[0].detach().float().cpu()
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    audio = _normalize_audio(audio, config.NORMALIZE_DBFS).clamp_(-1.0, 1.0)
    return audio, int(model.sampling_rate), gen_s


# ---------------------------------------------------------------- encoding
def encode(audio: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format {fmt!r}; use one of {', '.join(FORMATS)}")
    pcm16 = (audio.numpy() * 32767.0).astype(np.int16)
    if fmt == "pcm":
        return pcm16.tobytes()
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, pcm16, sample_rate, format=fmt.upper(), subtype="PCM_16")
        return buf.getvalue()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"ffmpeg not found on PATH; needed to encode {fmt}")
    codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
             "opus": ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"]}[fmt]
    proc = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
         "-i", "pipe:0", *codec, "pipe:1"],
        input=pcm16.tobytes(), capture_output=True, check=True,
    )
    return proc.stdout
