# vad.py
import numpy as np, soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps

MODEL = load_silero_vad()

class AudioValidationError(ValueError):
    """Audio de entrada inválido (formato, sample rate o canales incorrectos)."""

def recompute_turns(wav_source, sr_expected=8000):
    """wav_source: path o file-like (p.ej. io.BytesIO) con WAV estéreo."""
    try:
        audio, sr = sf.read(wav_source, dtype="float32")
    except Exception as e:
        raise AudioValidationError(f"no se pudo leer el WAV: {e}") from e

    if sr != sr_expected:
        raise AudioValidationError(f"esperaba {sr_expected} Hz, llegó {sr} Hz")
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise AudioValidationError("el archivo debe ser estéreo (2 canales)")

    duration_s = audio.shape[0] / sr

    turns = []
    for ch in (0, 1):
        segs = get_speech_timestamps(
            audio[:, ch],
            MODEL,
            sampling_rate=sr,
            threshold=0.5,
            min_speech_duration_ms=200,
            min_silence_duration_ms=350,
            speech_pad_ms=0,
            return_seconds=True,
        )
        for s in segs:
            turns.append({"channel": ch, "start": s["start"], "end": s["end"]})

    turns.sort(key=lambda t: t["start"])
    return {"turns": turns, "duration_s": duration_s}