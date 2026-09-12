# predict.py
import sys

import joblib
import pandas as pd

from vad import recompute_turns, AudioValidationError
from features_turns import turn_features, add_duration_features

MODEL_PATH = "detector.pkl"


def predict(wav_path):
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["cols"]

    vad_result = recompute_turns(wav_path)
    feats = turn_features(vad_result["turns"])
    feats = add_duration_features(feats, vad_result["duration_s"])

    row = pd.DataFrame([[feats.get(c, 0.0) for c in cols]], columns=cols)
    proba = float(model.predict_proba(row)[0, 1])
    return proba > 0.5, proba


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("uso: python predict.py ruta/al/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    try:
        is_synthetic, confidence = predict(wav_path)
    except AudioValidationError as e:
        print(f"audio inválido: {e}")
        sys.exit(1)

    etiqueta = "SINTÉTICO (IA)" if is_synthetic else "HUMANO"
    print(f"{wav_path}")
    print(f"  -> {etiqueta}  (confianza: {confidence:.4f})")
