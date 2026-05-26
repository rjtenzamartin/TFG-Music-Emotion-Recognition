"""
train_baseline.py -- Entrenamiento del modelo Baseline (SimpleCNN)
===================================================================
TFG: Reconocimiento de Emociones Musicales

Hiperparametros:
  - Epochs: 5
  - Batch size: 16
  - LR: 0.001 (Adam)
  - Loss: BCEWithLogitsLoss con pos_weight
  - Metrica: ROC-AUC (macro, global)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Rutas ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_utils import AudioPreprocessor
from dataset import MTGJamendoDataset
from models import SimpleCNN

# =====================================================================
# Configuracion
# =====================================================================
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.001
VAL_SPLIT = 0.2
SEED = 42

AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
PILOT_TSV = PROJECT_ROOT / "data" / "pilot_tracks.tsv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. Cargar datos
# =====================================================================
print("=" * 65)
print("  TRAIN BASELINE -- SimpleCNN (Piloto 200 tracks)")
print("=" * 65)
print(f"  Device: {DEVICE}")

# Cargar lista piloto
pilot_df = pd.read_csv(PILOT_TSV, sep="\t")
# Parsear tags (vienen como string de lista)
import ast
pilot_df["tags"] = pilot_df["tags"].apply(ast.literal_eval)

# Filtrar solo tracks que se descargaron correctamente
valid_mask = pilot_df["path"].apply(
    lambda p: (AUDIO_DIR / p).exists() and (AUDIO_DIR / p).stat().st_size > 10000
)
pilot_df = pilot_df[valid_mask].reset_index(drop=True)
print(f"  Tracks con audio valido: {len(pilot_df)}")

# Etiquetas (las 59 del dataset completo)
TSV_FULL = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
all_tags = set()
with open(TSV_FULL, encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        for t in parts[5:]:
            if t.startswith("mood/theme---"):
                all_tags.add(t.replace("mood/theme---", ""))
label_list = sorted(all_tags)
n_classes = len(label_list)
print(f"  Clases: {n_classes}")

# =====================================================================
# 2. Dataset y split
# =====================================================================
preprocessor = AudioPreprocessor(target_sr=32000, duration=30.0)
full_dataset = MTGJamendoDataset(
    df=pilot_df,
    audio_dir=AUDIO_DIR,
    preprocessor=preprocessor,
    label_list=label_list,
)

# Calcular pos_weight sobre el dataset COMPLETO (18k tracks) para pesos mas estables
full_records = []
with open(TSV_FULL, encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        tags = [t.replace("mood/theme---", "") for t in parts[5:] if t.startswith("mood/theme---")]
        full_records.append({"path": parts[3], "tags": tags})
full_df_for_weights = pd.DataFrame(full_records)
weight_dataset = MTGJamendoDataset(full_df_for_weights, ".", preprocessor, label_list)
pos_weight = weight_dataset.get_pos_weight().to(DEVICE)
print(f"  pos_weight: rango [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

# 80/20 split
n_val = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
torch.manual_seed(SEED)
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
print(f"  Train: {n_train} | Val: {n_val}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =====================================================================
# 3. Modelo, Loss, Optimizer
# =====================================================================
model = SimpleCNN(n_classes=n_classes, in_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Modelo: SimpleCNN ({total_params:,} params)")
print(f"  Loss: BCEWithLogitsLoss (pos_weight)")
print(f"  Optimizer: Adam (lr={LR})")
print(f"\n{'=' * 65}")

# =====================================================================
# 4. Training Loop
# =====================================================================
history = {"train_loss": [], "val_loss": [], "val_auc": []}


def evaluate(model, loader, criterion):
    """Evalua modelo y calcula loss + ROC-AUC."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ROC-AUC: solo para clases que tienen al menos un positivo y un negativo
    try:
        # Filtrar columnas con varianza
        valid_cols = []
        for j in range(all_labels.shape[1]):
            if all_labels[:, j].sum() > 0 and all_labels[:, j].sum() < len(all_labels):
                valid_cols.append(j)
        if len(valid_cols) > 0:
            auc = roc_auc_score(
                all_labels[:, valid_cols],
                all_logits[:, valid_cols],
                average="macro",
            )
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    return avg_loss, auc


t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    train_loss = running_loss / max(n_batches, 1)
    val_loss, val_auc = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)

    elapsed = time.time() - t0
    print(
        f"  Epoch {epoch}/{EPOCHS}  |  "
        f"Train Loss: {train_loss:.4f}  |  "
        f"Val Loss: {val_loss:.4f}  |  "
        f"Val AUC: {val_auc:.4f}  |  "
        f"Time: {elapsed:.0f}s"
    )

# =====================================================================
# 5. Guardar modelo
# =====================================================================
model_path = MODEL_DIR / "baseline_pilot.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "label_list": label_list,
    "n_classes": n_classes,
    "epochs": EPOCHS,
    "history": history,
}, model_path)
print(f"\n  Modelo guardado: {model_path}")

# =====================================================================
# 6. Grafica de curvas de entrenamiento
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, EPOCHS + 1)

# Loss
ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#2196F3")
ax1.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#FF5722")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCEWithLogitsLoss")
ax1.set_title("Loss (Train vs Val)", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# AUC
ax2.plot(epochs_range, history["val_auc"], "D-", label="Val ROC-AUC", color="#4CAF50", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("ROC-AUC (macro)")
ax2.set_title("Validation ROC-AUC", fontweight="bold")
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Baseline Pilot (SimpleCNN, 200 tracks, 5 epochs)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "training_curves.png", bbox_inches="tight", dpi=150)
print(f"  Curvas guardadas: {FIGURES_DIR / 'training_curves.png'}")

print(f"\n{'=' * 65}")
print(f"  ENTRENAMIENTO COMPLETADO")
print(f"  Mejor Val AUC: {max(history['val_auc']):.4f}")
print(f"  Tiempo total: {time.time() - t0:.0f}s")
print(f"{'=' * 65}")
