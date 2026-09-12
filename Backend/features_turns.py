# features_turns.py
import json, numpy as np

def _stats(xs, prefix):
    if len(xs) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean","std","min","max","med","n")}
    a = np.array(xs, dtype=float)
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std":  float(a.std()),
        f"{prefix}_min":  float(a.min()),
        f"{prefix}_max":  float(a.max()),
        f"{prefix}_med":  float(np.median(a)),
        f"{prefix}_n":    float(len(a)),
    }

PER_MIN_KEYS = ("caller_turns", "agent_turns", "interrupted_n",
                "silence_broken_by_caller", "gap_n", "resp_lat_n")

def turn_features(turns):
    """turns: lista de {"channel": 0|1, "start": float, "end": float}."""
    caller = sorted([t for t in turns if t["channel"] == 0], key=lambda t: t["start"])
    agent  = sorted([t for t in turns if t["channel"] == 1], key=lambda t: t["start"])

    f = {}

    # --- 1. LATENCIA DE RESPUESTA ---
    # tras cada turno del agente, cuánto tarda el caller en arrancar
    lat = []
    for a in agent:
        nxt = [c for c in caller if c["start"] >= a["end"]]
        if nxt:
            d = nxt[0]["start"] - a["end"]
            if d < 10:                      # descarta huecos de flujo
                lat.append(d)
    f.update(_stats(lat, "resp_lat"))
    # la VARIANZA suele discriminar más que la media
    f["resp_lat_cv"] = f["resp_lat_std"] / (f["resp_lat_mean"] + 1e-6)

    # --- 2. SOLAPAMIENTO / BARGE-IN ---
    overlap_total, barge_in = 0.0, 0
    for c in caller:
        for a in agent:
            ov = min(c["end"], a["end"]) - max(c["start"], a["start"])
            if ov > 0:
                overlap_total += ov
                if a["start"] > c["start"]:   # agente entró encima del caller
                    barge_in += 1
    caller_speech = sum(c["end"] - c["start"] for c in caller) or 1e-6
    f["overlap_ratio"]   = overlap_total / caller_speech
    f["interrupted_n"]   = float(barge_in)

    # --- 3. RECUPERACIÓN TRAS INTERRUPCIÓN ---
    # cuando el agente habla encima, ¿cuánto tarda el caller en callarse?
    stop_delay = []
    for a in agent:
        for c in caller:
            if c["start"] < a["start"] < c["end"]:
                stop_delay.append(c["end"] - a["start"])
    f.update(_stats(stop_delay, "stop_delay"))

    # --- 4. RESPUESTA AL SILENCIO ---
    # huecos donde nadie habla: ¿quién los rompe y en cuánto?
    gaps = []
    events = sorted(turns, key=lambda t: t["start"])
    for i in range(len(events) - 1):
        g = events[i+1]["start"] - events[i]["end"]
        if g > 1.0:
            gaps.append((g, events[i+1]["channel"]))
    f["silence_broken_by_caller"] = float(sum(1 for g, ch in gaps if ch == 0))
    f.update(_stats([g for g, _ in gaps], "gap"))

    # --- 5. FORMA DE LOS TURNOS ---
    f.update(_stats([c["end"] - c["start"] for c in caller], "caller_turn"))
    f["caller_turns"]  = float(len(caller))
    f["agent_turns"]   = float(len(agent))
    f["turn_ratio"]    = len(caller) / (len(agent) + 1e-6)

    return f

def turn_features_from_file(turns_path):
    with open(turns_path, encoding="utf-8") as fh:
        turns = json.load(fh)["turns"]
    return turn_features(turns)

def add_duration_features(feats, duration_s):
    """Normaliza por duración (anti-fuga): agrega 'dur' y las variantes _per_min.
    Debe usarse igual en entrenamiento (build_dataset.py) e inferencia (app.py)
    para evitar desajustes train/serve."""
    dur = max(float(duration_s), 1e-6)
    out = dict(feats)
    out["dur"] = dur
    for k in PER_MIN_KEYS:
        if k in out:
            out[f"{k}_per_min"] = out[k] / (dur / 60.0)
    return out