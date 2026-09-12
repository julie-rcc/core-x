# train.py
import pandas as pd, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss

EXCLUIR = {"anon_id", "y", "split", "dur"}   # 'dur' fuera: es fuga directa

df = pd.read_csv("features.csv")
cols = [c for c in df.columns if c not in EXCLUIR]

tr = df[df.split == "train"]
va = df[df.split == "val"]
print(f"train: {len(tr)} ({tr.y.mean():.1%} sintéticas)")
print(f"val:   {len(va)} ({va.y.mean():.1%} sintéticas)")
print(f"features: {len(cols)}\n")

base = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
)
base.fit(tr[cols], tr.y)
p = base.predict_proba(va[cols])[:, 1]

print("AUC val:  ", round(roc_auc_score(va.y, p), 4))
print("Brier val:", round(brier_score_loss(va.y, p), 4), "\n")
print(classification_report(va.y, p > 0.5, target_names=["human", "synthetic"]))

print("--- top features ---")
w = base[-1].coef_[0]
for name, coef in sorted(zip(cols, w), key=lambda x: -abs(x[1]))[:15]:
    signo = "sintético" if coef > 0 else "humano"
    print(f"  {coef:+.3f}  {name:<32} -> {signo}")

cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
cal.fit(tr[cols], tr.y)
p_cal = cal.predict_proba(va[cols])[:, 1]
print(f"\nBrier calibrado: {round(brier_score_loss(va.y, p_cal), 4)}")

joblib.dump({"model": cal, "cols": cols}, "detector.pkl")
print("guardado -> detector.pkl")