# app.py
import base64
import io
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from vad import recompute_turns, AudioValidationError
from features_turns import turn_features, add_duration_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector.pkl")

# Nombres de campo aceptados para el WAV en base64: el README del challenge
# no especifica la clave exacta del JSON, así que aceptamos varias comunes.
AUDIO_FIELD_NAMES = ("audio_base64", "audio", "wav_base64", "wav", "audio_wav_base64")

app = FastAPI(title="Human vs Synthetic Caller Detector")

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"no se encontró {MODEL_PATH}; corre train.py primero")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.on_event("startup")
def _load_model_at_startup():
    get_bundle()


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="el body debe ser JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="el body debe ser un objeto JSON")

    b64 = next((body[k] for k in AUDIO_FIELD_NAMES if body.get(k)), None)
    if b64 is None:
        raise HTTPException(
            status_code=400,
            detail=f"falta el audio en base64; se espera uno de estos campos: {AUDIO_FIELD_NAMES}",
        )

    try:
        wav_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="el campo de audio no es base64 válido")

    try:
        vad_result = recompute_turns(io.BytesIO(wav_bytes))
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    feats = turn_features(vad_result["turns"])
    feats = add_duration_features(feats, vad_result["duration_s"])

    bundle = get_bundle()
    model, cols = bundle["model"], bundle["cols"]
    row = pd.DataFrame([[feats.get(c, 0.0) for c in cols]], columns=cols)

    proba = float(model.predict_proba(row)[0, 1])
    return DetectResponse(is_synthetic=proba > 0.5, confidence=round(proba, 4))
