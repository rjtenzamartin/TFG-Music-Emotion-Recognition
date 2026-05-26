"""
Runner script for evaluation notebook cells (non-interactive).
Executes the key cells from notebooks/2_evaluacion_piloto.ipynb.
"""
import sys, random, ast, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_utils import AudioPreprocessor
from models import SimpleCNN

# =====================================================================
# 1. Curvas de entrenamiento (tabla)
# =====================================================================
model_path = PROJECT_ROOT / "models" / "baseline_pilot.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

history = checkpoint["history"]
print("=" * 55)
print("  CURVAS DE ENTRENAMIENTO")
print("=" * 55)
print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>9}")
print(f"  {'-'*5}  {'-'*11}  {'-'*9}  {'-'*9}")
for i in range(len(history["train_loss"])):
    print(f"  {i+1:>5}  {history['train_loss'][i]:>11.4f}  {history['val_loss'][i]:>9.4f}  {history['val_auc'][i]:>9.4f}")
print(f"\n  Mejor Val AUC: {max(history['val_auc']):.4f}")

# =====================================================================
# 2. Cargar Modelo
# =====================================================================
label_list = checkpoint["label_list"]
n_classes = checkpoint["n_classes"]

model = SimpleCNN(n_classes=n_classes, in_channels=1)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Modelo cargado: SimpleCNN ({total_params:,} params)")

# =====================================================================
# 3. Inferencia
# =====================================================================
pilot_df = pd.read_csv(PROJECT_ROOT / "data" / "pilot_tracks.tsv", sep="\t")
pilot_df["tags"] = pilot_df["tags"].apply(ast.literal_eval)
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
pilot_df = pilot_df[
    pilot_df["path"].apply(lambda p: (AUDIO_DIR / p).exists())
].reset_index(drop=True)

random.seed(99)
idx = random.randint(0, len(pilot_df) - 1)
row = pilot_df.iloc[idx]

audio_path = AUDIO_DIR / row["path"]
real_tags = row["tags"]

print(f"\n{'=' * 55}")
print(f"  INFERENCIA DE PRUEBA")
print(f"{'=' * 55}")
print(f"  Track: {row['path']}")
print(f"  ID:    {row['track_id']}")
print(f"  Real:  {real_tags}")

preprocessor = AudioPreprocessor(target_sr=32000, duration=30.0)
y = preprocessor.load_audio(str(audio_path))
y = preprocessor._fix_length(y)
y = preprocessor._normalize(y)
log_mel = preprocessor.get_melspec(y)

x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(f"  Input: {list(x.shape)}")

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze()

top10_idx = torch.argsort(probs, descending=True)[:10]

print(f"\n  {'#':>3}  {'Etiqueta':<22}  {'Prob':>7}  {'Real':>6}")
print(f"  {'-'*3}  {'-'*22}  {'-'*7}  {'-'*6}")
for rank, i in enumerate(top10_idx, 1):
    tag = label_list[i]
    prob = probs[i].item()
    is_real = "SI" if tag in real_tags else ""
    marker = " <<" if tag in real_tags else ""
    print(f"  {rank:>3}  {tag:<22}  {prob:>6.1%}  {is_real:>6}{marker}")

hits = sum(1 for t in real_tags if t in [label_list[i] for i in top10_idx])
print(f"\n  Aciertos en top-10: {hits}/{len(real_tags)}")
print(f"\n{'=' * 55}")
print(f"  EVALUACION COMPLETA")
print(f"{'=' * 55}")
