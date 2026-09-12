# inspect.py
import pandas as pd, os

m = pd.read_csv("manifest.csv")
print(m.head(), "\n")
print("filas:", len(m))
print("\ncolumnas:", list(m.columns))

print("\n--- balance de clases ---")
print(pd.crosstab(m.split, m.label))

print("\n--- duración por clase (LA TRAMPA) ---")
print(m.groupby("label").duration_s.describe())

print("\n--- archivos faltantes ---")
falta_wav  = [i for i in m.anon_id if not os.path.exists(f"audio/{i}.wav")]
falta_json = [i for i in m.anon_id if not os.path.exists(f"turns/{i}.json")]
print("sin wav:", len(falta_wav), "| sin json:", len(falta_json))
if falta_wav[:3]:  print("ejemplos:", falta_wav[:3])