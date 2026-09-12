# core-x — Detector de llamante humano vs. sintético

Solución para el reto de Altur en HackMTY 2026: dado el audio de una llamada
a un agente bancario de IA, decidir si quien llama es una persona real o un
caller sintético.

> Todo el código vive en [`Backend/`](Backend/).

## Cómo funciona

En vez de analizar el audio crudo (espectro, timbre de la voz), el modelo usa
**la dinámica de la conversación**: qué tan rápido y consistente responde el
llamante, cuándo interrumpe, cómo reacciona a los silencios. Esta señal es
más difícil de imitar para un sistema sintético y generaliza mejor a voces
nunca antes vistas (justo el criterio de evaluación del challenge).

Pipeline:

```
audio WAV (8kHz, estéreo) -> VAD (vad.py) -> turnos de habla por canal
                           -> features de timing (features_turns.py)
                           -> clasificador calibrado (detector.pkl)
                           -> {is_synthetic, confidence}
```

- **Canal 0** = llamante (lo que se clasifica). **Canal 1** = agente.
- El modelo es una regresión logística calibrada (`CalibratedClassifierCV`),
  entrenada sobre 353 llamadas (train/val separados por speaker, sin
  solapamiento). AUC de validación: **0.994** (single split) / **0.986 ± 0.011**
  (5-fold CV sobre todo el dataset) — ver [Pendientes](#pendientes-antes-de-la-entrega)
  sobre el riesgo de que esto no generalice al set oculto del challenge.

## Estructura del repo

Todo dentro de [`Backend/`](Backend/):

| Archivo | Qué hace |
|---|---|
| `app.py` | Servidor FastAPI — expone `POST /detect` y `GET /health` |
| `predict.py` | CLI para probar un WAV local sin levantar el servidor |
| `vad.py` | Detección de turnos de habla por canal (Silero VAD, backend ONNX) |
| `features_turns.py` | Extracción de features de timing conversacional |
| `build_dataset.py` | Genera `features.csv` a partir de `manifest.csv` + `turns/` |
| `train.py` | Entrena y calibra el modelo -> `detector.pkl` |
| `inspect_data.py` | Inspección rápida del dataset (balance de clases, archivos faltantes) |
| `manifest.csv`, `turns/` | Dataset del challenge (labels + turnos precomputados) |
| `detector.pkl` | Modelo entrenado, listo para servir |
| `Dockerfile`, `.dockerignore` | Imagen del servidor |

`Backend/audio/` (los WAV crudos, ~1.6GB) está en `.gitignore` — se descarga
aparte desde el release `altur-challenge-audio.zip` del repo del challenge.

## Cómo correrlo

### Local

```bash
cd Backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Probar un audio suelto sin servidor:

```bash
cd Backend
python predict.py audio/algun_call.wav
```

### Docker

```bash
cd Backend
docker build -t core-x-detector .
docker run -p 8000:8000 core-x-detector
```

> No se pudo probar el build de Docker en esta máquina (no hay Docker
> instalado) — verificar antes de depender de esto para la demo.

### Reentrenar el modelo

```bash
cd Backend
python build_dataset.py   # manifest.csv + turns/ -> features.csv
python train.py           # features.csv -> detector.pkl
```

## API

### `POST /detect`

Request (JSON):
```json
{ "audio_base64": "<WAV estéreo 8kHz codificado en base64>" }
```

> El campo exacto esperado por los organizadores **no está confirmado**
> (ver [Pendientes](#pendientes-antes-de-la-entrega)). Hoy el endpoint acepta
> varios alias por seguridad: `audio_base64`, `audio`, `wav_base64`, `wav`,
> `audio_wav_base64`.

Response (200):
```json
{ "is_synthetic": true, "confidence": 0.9989 }
```

Errores (400) con `{"detail": "..."}` para: falta el campo de audio, base64
inválido, WAV corrupto/formato no reconocido, sample rate distinto a 8kHz,
o audio no estéreo.

### `GET /health`

```json
{ "status": "ok" }
```

## Pendientes antes de la entrega

- [ ] **Confirmar con los organizadores** el nombre exacto del campo JSON
      del audio, y cómo van a invocar el endpoint (¿URL pública?, ¿Docker?,
      ¿límite de tiempo de respuesta?).
- [ ] Probar el build de Docker (no disponible en la máquina donde se
      desarrolló esto).
- [ ] El AUC (~0.99) se validó contra el propio dataset de entrenamiento; no
      hay garantía de que se sostenga si el sintetizador de voz del set de
      evaluación oculto es distinto al usado en estas 353 llamadas.
