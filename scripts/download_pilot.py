"""
download_pilot.py -- Descarga un subset piloto de 200 canciones de MTG-Jamendo
================================================================================
TFG: Reconocimiento de Emociones Musicales

Usa la API de descarga de Jamendo:
  https://prod-1.storage.jamendo.com/download/track/{TRACK_ID}/mp32/
"""

import os
import sys
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuracion ──
N_PILOT = 200
SEED = 42
JAMENDO_BASE = "https://prod-1.storage.jamendo.com/download/track/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TSV_PATH = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

# ── Cargar TSV ──
print("=" * 60)
print("  DOWNLOAD PILOT -- 200 tracks de MTG-Jamendo")
print("=" * 60)

records = []
with open(TSV_PATH, encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        track_id = parts[0]
        path = parts[3]
        tags = [t.replace("mood/theme---", "") for t in parts[5:] if t.startswith("mood/theme---")]
        records.append({"track_id": track_id, "path": path, "tags": tags})

df = pd.DataFrame(records)
print(f"  Total tracks en TSV: {len(df):,}")

# ── Seleccionar 200 aleatorios ──
random.seed(SEED)
pilot_idx = random.sample(range(len(df)), min(N_PILOT, len(df)))
pilot_df = df.iloc[pilot_idx].reset_index(drop=True)
print(f"  Seleccionados: {len(pilot_df)}")

# ── Crear directorio ──
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ── Guardar lista de tracks seleccionados ──
pilot_tsv = PROJECT_ROOT / "data" / "pilot_tracks.tsv"
pilot_df.to_csv(pilot_tsv, sep="\t", index=False)
print(f"  Lista guardada: {pilot_tsv}")

# ── Descargar ──
print(f"\n  Descargando a: {AUDIO_DIR}")
print("-" * 60)

ok, fail, skip = 0, 0, 0
for _, row in tqdm(pilot_df.iterrows(), total=len(pilot_df), desc="Descargando"):
    # El path es como "56/1376256.mp3" -> track_numeric_id = "1376256"
    track_path = row["path"]
    track_numeric_id = Path(track_path).stem  # "1376256" from "56/1376256.mp3"

    # Guardar con la estructura de carpetas del path original
    out_path = AUDIO_DIR / track_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10000:
        skip += 1
        continue

    url = f"{JAMENDO_BASE}{track_numeric_id}/mp32/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        if out_path.stat().st_size > 10000:
            ok += 1
        else:
            fail += 1
            out_path.unlink(missing_ok=True)
    except Exception as e:
        fail += 1
        tqdm.write(f"  FAIL: {track_numeric_id} -> {e}")

print(f"\n{'=' * 60}")
print(f"  RESULTADO")
print(f"{'=' * 60}")
print(f"  Descargados:  {ok:>5}")
print(f"  Ya existian:  {skip:>5}")
print(f"  Fallidos:     {fail:>5}")
print(f"  Total audio:  {ok + skip:>5} / {len(pilot_df)}")
print(f"{'=' * 60}")
