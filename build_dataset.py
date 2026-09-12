# build_dataset.py
import os, json, time
import pandas as pd
from features_turns import turn_features_from_file, add_duration_features

MANIFEST = "manifest.csv"
TURNS_DIR = "turns"
OUT = "features.csv"

def build(use_recomputed_vad=False):
    m = pd.read_csv(MANIFEST)
    rows, errores = [], []
    t0 = time.time()

    for n, r in enumerate(m.itertuples(index=False), 1):
        path = os.path.join(TURNS_DIR, f"{r.anon_id}.json")
        if not os.path.exists(path):
            errores.append((r.anon_id, "json faltante"))
            continue

        try:
            feats = turn_features_from_file(path)
        except Exception as e:
            errores.append((r.anon_id, f"{type(e).__name__}: {e}"))
            continue

        feats = add_duration_features(feats, r.duration_s)
        feats["anon_id"] = r.anon_id
        feats["y"] = int(r.label == "synthetic")
        feats["split"] = r.split
        rows.append(feats)

        if n % 50 == 0:
            print(f"  {n}/{len(m)}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUT, index=False)

    print(f"\nlisto: {len(df)} filas, {df.shape[1]} columnas -> {OUT}")
    print(f"tiempo: {time.time()-t0:.1f}s")
    if errores:
        print(f"\n{len(errores)} errores:")
        for a, e in errores[:5]:
            print(f"  {a}: {e}")
    return df

if __name__ == "__main__":
    build()
    