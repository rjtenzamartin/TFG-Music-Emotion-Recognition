"""
check_dataloader.py — Verificación rápida del Dataset y Modelo
===============================================================
TFG: Reconocimiento de Emociones Musicales

Crea datos sintéticos (5 audios fake), instancia el Dataset y el
DataLoader, y verifica las dimensiones del batch.
"""

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Rutas ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from audio_utils import AudioPreprocessor
from dataset import MTGJamendoDataset
from models import SimpleCNN

# =====================================================================
# 1. Crear datos sintéticos (5 audios fake de 32 kHz × 30 s)
# =====================================================================
print("=" * 60)
print("  CHECK DATALOADER — Verificación de dimensiones")
print("=" * 60)

SR = 32_000
DURATION = 30.0
N_FAKE = 5

# Etiquetas de ejemplo (subset de las 59 reales)
FAKE_LABELS = [
    "happy", "sad", "energetic", "relaxing", "dark",
    "epic", "romantic", "melancholic", "upbeat", "calm",
]

# Simular las 59 etiquetas completas
TSV_PATH = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
if TSV_PATH.exists():
    with open(TSV_PATH, encoding="utf-8") as f:
        f.readline()
        all_tags = set()
        for line in f:
            parts = line.strip().split("\t")
            for t in parts[5:]:
                if t.startswith("mood/theme---"):
                    all_tags.add(t.replace("mood/theme---", ""))
    label_list = sorted(all_tags)
    print(f"  Etiquetas cargadas del TSV: {len(label_list)}")
else:
    label_list = sorted(FAKE_LABELS + [f"tag_{i}" for i in range(49)])
    print(f"  Etiquetas sintéticas: {len(label_list)}")

n_classes = len(label_list)

# Crear audios sintéticos en directorio temporal
tmpdir = tempfile.mkdtemp(prefix="tfg_check_")
print(f"  Audio temporal: {tmpdir}")

import soundfile as sf

records = []
for i in range(N_FAKE):
    # Generar señal aleatoria
    y = np.random.randn(int(SR * DURATION)).astype(np.float32) * 0.1
    fname = f"fake_{i:03d}.wav"
    fpath = Path(tmpdir) / fname
    sf.write(str(fpath), y, SR)

    # Asignar 2-3 tags aleatorios
    n_tags = np.random.randint(2, 4)
    tags = list(np.random.choice(label_list[:10], size=n_tags, replace=False))
    records.append({"path": fname, "tags": tags})

df = pd.DataFrame(records)
print(f"\n  DataFrame fake ({len(df)} filas):")
for _, row in df.iterrows():
    print(f"    {row['path']} -> {row['tags']}")

# =====================================================================
# 2. Instanciar Dataset y DataLoader
# =====================================================================
preprocessor = AudioPreprocessor(target_sr=SR, duration=DURATION)
dataset = MTGJamendoDataset(
    df=df,
    audio_dir=tmpdir,
    preprocessor=preprocessor,
    label_list=label_list,
)

print(f"\n  Dataset: {len(dataset)} samples")

loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

# =====================================================================
# 3. Obtener un batch y mostrar dimensiones
# =====================================================================
batch_spectrograms, batch_labels = next(iter(loader))

print(f"\n{'=' * 60}")
print(f"  BATCH SHAPES")
print(f"{'=' * 60}")
print(f"  Input  (spectrograms): {list(batch_spectrograms.shape)}")
print(f"         Esperado:       [2, 1, {preprocessor.n_mels}, T]")
print(f"  Target (labels):       {list(batch_labels.shape)}")
print(f"         Esperado:       [2, {n_classes}]")
print(f"  Input dtype:           {batch_spectrograms.dtype}")
print(f"  Target dtype:          {batch_labels.dtype}")
print(f"  Input rango:           [{batch_spectrograms.min():.1f}, {batch_spectrograms.max():.1f}]")
print(f"  Target sum por sample: {batch_labels.sum(dim=1).tolist()}")

# =====================================================================
# 4. Pasar por el modelo SimpleCNN
# =====================================================================
model = SimpleCNN(n_classes=n_classes, in_channels=1)
model.eval()

with torch.no_grad():
    logits = model(batch_spectrograms)

print(f"\n{'=' * 60}")
print(f"  MODEL FORWARD PASS")
print(f"{'=' * 60}")
print(f"  Modelo:    SimpleCNN")
print(f"  Input:     {list(batch_spectrograms.shape)}")
print(f"  Output:    {list(logits.shape)}")
print(f"             Esperado: [2, {n_classes}]")
print(f"  Logits[0]: [{logits[0].min():.3f}, {logits[0].max():.3f}]")

# ── Parámetros totales ──
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Parámetros totales:     {total_params:>10,}")
print(f"  Parámetros entrenables: {trainable:>10,}")

# ── Loss de ejemplo ──
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(logits, batch_labels)
print(f"\n  BCEWithLogitsLoss:      {loss.item():.4f}")

print(f"\n{'=' * 60}")
print(f"  ✅ TODAS LAS VERIFICACIONES PASADAS")
print(f"{'=' * 60}")

# Limpiar
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
