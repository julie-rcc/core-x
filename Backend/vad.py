# vad.py
from concurrent.futures import ThreadPoolExecutor

import numpy as np, soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps

# Backend ONNX: cada sesión corre en 1 solo hilo (config interna de silero-vad),
# lo que la hace más rápida en CPU y segura para paralelizar entre canales.
# El modelo guarda estado interno (RNN) que se resetea en cada llamada
# (ver get_speech_timestamps -> model.reset_states()), por eso NO se puede
# compartir una misma instancia entre llamadas concurrentes: se necesita
# una instancia independiente por canal.
_MODEL_CH0 = load_silero_vad(onnx=True)
_MODEL_CH1 = load_silero_vad(onnx=True)
_POOL = ThreadPoolExecutor(max_workers=2)


def _warmup():
    dummy = np.zeros(8000, dtype="float32")  # 1s de silencio a 8kHz
    for model in (_MODEL_CH0, _MODEL_CH1):
        get_speech_timestamps(dummy, model, sampling_rate=8000)


_warmup()


class AudioValidationError(ValueError):
    """Audio de entrada inválido (formato, sample rate o canales incorrectos)."""


def _vad_channel(audio_ch, sr, model):
    segs = get_speech_timestamps(
        audio_ch,
        model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=200,
        min_silence_duration_ms=350,
        speech_pad_ms=0,
        return_seconds=True,
    )
    return segs


def recompute_turns(wav_source, sr_expected=8000):
    """wav_source: path o file-like (p.ej. io.BytesIO) con WAV estéreo."""
    try:
        audio, sr = sf.read(wav_source, dtype="float32")
    except Exception as e:
        raise AudioValidationError(
            "no se pudo leer el archivo como WAV (formato inválido o corrupto)"
        ) from e

    if sr != sr_expected:
        raise AudioValidationError(f"esperaba {sr_expected} Hz, llegó {sr} Hz")
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise AudioValidationError("el archivo debe ser estéreo (2 canales)")

    duration_s = audio.shape[0] / sr

    fut0 = _POOL.submit(_vad_channel, audio[:, 0], sr, _MODEL_CH0)
    fut1 = _POOL.submit(_vad_channel, audio[:, 1], sr, _MODEL_CH1)

    turns = []
    for ch, fut in ((0, fut0), (1, fut1)):
        for s in fut.result():
            turns.append({"channel": ch, "start": s["start"], "end": s["end"]})

    turns.sort(key=lambda t: t["start"])
    return {"turns": turns, "duration_s": duration_s}