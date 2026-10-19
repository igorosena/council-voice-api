"""Council Voice API — a small headless voice-cloning TTS service.

    POST /voices            register a cloned voice (multipart)
    GET  /voices            list voices
    DELETE /voices/{id}     remove a voice
    POST /speak             {voice, text, ...} -> audio bytes
    POST /v1/audio/speech   OpenAI-compatible alias of /speak
    GET  /health
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import config, engine, voices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("council")


# ---------------------------------------------------------------- auth
async def require_key(request: Request) -> None:
    if not config.API_KEY:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    token = token or request.headers.get("x-api-key", "").strip()
    if token != config.API_KEY:
        raise HTTPException(401, "invalid or missing API key")


# ---------------------------------------------------------------- lifecycle
def _idle_timer() -> None:
    while True:
        threading.Event().wait(60)
        try:
            engine.unload_if_idle()
        except Exception:  # pragma: no cover
            logger.exception("idle unload failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (config.REFS_DIR, config.PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if config.IDLE_UNLOAD_SECONDS:
        threading.Thread(target=_idle_timer, daemon=True).start()
    if config.PRELOAD:
        await asyncio.to_thread(engine.get_model)
    yield


app = FastAPI(title="Council Voice API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------- health
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.MODEL_ID,
        "model_loaded": engine.is_loaded(),
        "device": engine.resolve_device(),
        "voices": len(voices.list_voices()),
    }


# ---------------------------------------------------------------- voices
@app.get("/voices", dependencies=[Depends(require_key)])
async def list_voices():
    return {"voices": [v.public() for v in voices.list_voices()]}


@app.post("/voices", dependencies=[Depends(require_key)], status_code=201)
async def create_voice(
    id: str = Form(..., description="slug used in /speak, e.g. 'igor'"),
    ref_audio: UploadFile = File(..., description="3-15 s clean speech clip"),
    ref_text: str = Form(..., description="exact transcript of the clip"),
    name: Optional[str] = Form(None),
    language: str = Form(config.DEFAULT_LANGUAGE),
):
    voice_id = voices.validate_id(id)
    ref_text = ref_text.strip()
    if not ref_text:
        raise HTTPException(400, "ref_text is required (the exact transcript of the clip)")

    suffix = Path(ref_audio.filename or "").suffix or ".bin"
    tmp_upload = config.REFS_DIR / f".{voice_id}-{uuid.uuid4().hex}{suffix}"
    tmp_upload.write_bytes(await ref_audio.read())
    wav_path = config.REFS_DIR / f"{voice_id}.wav"
    prompt_path = config.PROMPTS_DIR / f"{voice_id}.pt"
    try:
        try:
            await asyncio.to_thread(engine.convert_to_wav, tmp_upload, wav_path)
        except Exception as e:
            raise HTTPException(400, f"could not decode reference audio: {e}") from e
        seconds = engine.probe_seconds(wav_path)
        if seconds < 1.0:
            raise HTTPException(400, f"reference clip too short ({seconds:.1f}s); use 3-15 s")
        if seconds > config.MAX_REF_SECONDS:
            raise HTTPException(
                400,
                f"reference clip is {seconds:.1f}s; trim it to at most "
                f"{config.MAX_REF_SECONDS:.0f}s so it matches ref_text",
            )
        cleaned = await asyncio.to_thread(engine.clean_reference, wav_path)
        await asyncio.to_thread(engine.encode_prompt, wav_path, ref_text, prompt_path)
    finally:
        tmp_upload.unlink(missing_ok=True)

    voice = voices.Voice(
        id=voice_id,
        name=(name or voice_id).strip(),
        language=language.strip() or config.DEFAULT_LANGUAGE,
        ref_text=ref_text,
        ref_audio=str(wav_path.relative_to(config.DATA_DIR)),
        prompt=str(prompt_path.relative_to(config.DATA_DIR)),
        ref_seconds=round(seconds, 2),
        cleaned=cleaned,
    )
    voices.upsert(voice)
    logger.info("registered voice %s (%.1fs ref, cleaned=%s)", voice_id, seconds, cleaned)
    return voice.public()


@app.delete("/voices/{voice_id}", dependencies=[Depends(require_key)])
async def delete_voice(voice_id: str):
    if voices.delete(voice_id.lower()) is None:
        raise HTTPException(404, f"voice {voice_id!r} not found")
    return {"deleted": voice_id}


# ---------------------------------------------------------------- speak
class SpeakRequest(BaseModel):
    voice: str = Field(..., description="voice id registered via POST /voices")
    text: str = Field(..., min_length=1, max_length=config.MAX_TEXT_CHARS)
    language: Optional[str] = Field(None, description="'pt', 'Portuguese', 'en', ... or 'auto'")
    format: Literal["mp3", "wav", "opus", "flac", "pcm"] = config.DEFAULT_FORMAT  # type: ignore[assignment]
    speed: float = Field(1.0, ge=0.5, le=2.0)
    num_step: Optional[int] = Field(None, ge=4, le=64, description="diffusion steps; 16 fast, 32 best")
    guidance_scale: Optional[float] = Field(None, ge=0.5, le=5.0)
    seed: Optional[int] = None


async def _speak(req: SpeakRequest) -> Response:
    voice = voices.get_voice(req.voice)
    if voice is None:
        raise HTTPException(404, f"voice {req.voice!r} not found; see GET /voices")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    language = req.language or voice.language
    try:
        audio, sr, gen_s = await asyncio.to_thread(
            engine.synthesize, voice.prompt_path, text, language,
            req.speed, req.num_step, req.guidance_scale, req.seed,
        )
        payload = await asyncio.to_thread(engine.encode, audio, sr, req.format)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("synthesis failed for voice %s", voice.id)
        raise HTTPException(500, f"synthesis failed: {e}") from e

    duration = audio.numel() / sr
    mime, ext = engine.FORMATS[req.format]
    logger.info("spoke %s: %d chars -> %.1fs audio in %.1fs", voice.id, len(text), duration, gen_s)
    return Response(
        content=payload,
        media_type=mime,
        headers={
            "Content-Disposition": f'inline; filename="{voice.id}.{ext}"',
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Gen-Time": f"{gen_s:.3f}",
            "X-Sample-Rate": str(sr),
        },
    )


@app.post("/speak", dependencies=[Depends(require_key)])
async def speak(req: SpeakRequest):
    return await _speak(req)


class OpenAISpeechRequest(BaseModel):
    model: str = "omnivoice"
    input: str
    voice: str
    response_format: Literal["mp3", "wav", "opus", "flac", "pcm"] = "mp3"
    speed: float = 1.0
    language: Optional[str] = None
    seed: Optional[int] = None


@app.post("/v1/audio/speech", dependencies=[Depends(require_key)])
async def openai_speech(req: OpenAISpeechRequest):
    return await _speak(SpeakRequest(
        voice=req.voice, text=req.input, language=req.language,
        format=req.response_format, speed=max(0.5, min(2.0, req.speed)), seed=req.seed,
    ))


@app.get("/v1/audio/voices", dependencies=[Depends(require_key)])
async def openai_voices():
    return {"voices": [v.public() for v in voices.list_voices()]}
