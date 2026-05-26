"""
check_setup.py -- Verificacion del Dataset (pos_weight) y Modelo
=================================================================
TFG: Reconocimiento de Emociones Musicales
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from audio_utils import AudioPreprocessor
from dataset import MTGJamendoDataset
from models import SimpleCNN

# =====================================================================
# 1. Cargar el TSV real para calcular pos_weight
# =====================================================================
TSV_PATH = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"

records = []
with open(TSV_PATH, encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        tags = [t.replace("mood/theme---", "") for t in parts[5:] if t.startswith("mood/theme---")]
        records.append({"path": parts[3], "tags": tags})

df = pd.DataFrame(records)
label_list = sorted({t for tags in df["tags"] for t in tags})
n_classes = len(label_list)

print("=" * 60)
print("  CHECK SETUP -- Dataset + Modelo")
print("=" * 60)
print(f"  Tracks:     {len(df):,}")
print(f"  Etiquetas:  {n_classes}")

# =====================================================================
# 2. Instanciar Dataset y calcular pos_weight
# =====================================================================
preprocessor = AudioPreprocessor(target_sr=32000, duration=30.0)
dataset = MTGJamendoDataset(
    df=df,
    audio_dir=PROJECT_ROOT / "data",  # no se usa para pos_weight
    preprocessor=preprocessor,
    label_list=label_list,
)

pos_weight = dataset.get_pos_weight()

print(f"\n  pos_weight shape: {list(pos_weight.shape)}")
print(f"  pos_weight rango: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

# Top 5 etiquetas mas raras (peso mas alto)
sorted_idx = torch.argsort(pos_weight, descending=True)
print(f"\n  {'Etiqueta':<25s} {'Peso':>8s}  {'Positivos':>10s}  {'Significado'}")
print(f"  {'-'*25} {'-'*8}  {'-'*10}  {'-'*20}")
for rank, i in enumerate(sorted_idx[:5]):
    tag = label_list[i]
    w = pos_weight[i].item()
    n_pos = int(len(df) / (1 + w))  # P = N_total / (1 + w) approx
    print(f"  {tag:<25s} {w:>8.1f}  {n_pos:>10,}  etiqueta MUY rara")

print(f"\n  Top 3 etiquetas mas comunes (peso mas bajo):")
for rank, i in enumerate(sorted_idx[-3:]):
    tag = label_list[i]
    w = pos_weight[i].item()
    n_pos = int(len(df) / (1 + w))
    print(f"  {tag:<25s} {w:>8.1f}  {n_pos:>10,}")

# =====================================================================
# 3. Forward pass con datos sinteticos
# =====================================================================
import soundfile as sf
import tempfile, shutil

tmpdir = tempfile.mkdtemp(prefix="tfg_check_")
fake_records = []
for i in range(4):
    y = np.random.randn(int(32000 * 30)).astype(np.float32) * 0.1
    fname = f"fake_{i:03d}.wav"
    sf.write(str(Path(tmpdir) / fname), y, 32000)
    n_tags = np.random.randint(2, 4)
    tags = list(np.random.choice(label_list[:10], size=n_tags, replace=False))
    fake_records.append({"path": fname, "tags": tags})

fake_df = pd.DataFrame(fake_records)
fake_dataset = MTGJamendoDataset(fake_df, tmpdir, preprocessor, label_list)
loader = DataLoader(fake_dataset, batch_size=2, shuffle=False, num_workers=0)

batch_x, batch_y = next(iter(loader))

model = SimpleCNN(n_classes=n_classes, in_channels=1)
model.eval()
with torch.no_grad():
    logits = model(batch_x)

# Loss con pos_weight
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logits, batch_y)

print(f"\n{'=' * 60}")
print(f"  FORWARD PASS")
print(f"{'=' * 60}")
print(f"  Input:   {list(batch_x.shape)}")
print(f"  Output:  {list(logits.shape)}")
print(f"  Target:  {list(batch_y.shape)}")
print(f"  Loss (weighted): {loss.item():.4f}")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parametros:      {total_params:,}")

print(f"\n{'=' * 60}")
print(f"  TODAS LAS VERIFICACIONES PASADAS")
print(f"{'=' * 60}")

shutil.rmtree(tmpdir, ignore_errors=True)
