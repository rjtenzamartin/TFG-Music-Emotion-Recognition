"""
train_panns.py — Entrenamiento SOTA de PANNs (CNN14)
=====================================================
TFG: Reconocimiento de Emociones Musicales
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import random

# --- TUS MODULOS ---
from dataset import MTGJamendoDataset, load_split_tsv
from audio_utils import AudioPreprocessor
from models_panns import build_panns_model

def set_seed(seed: int = 42):
    print(f"Fijando semilla de reproducibilidad a: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

# LLAMAR A LA FUNCIÓN INMEDIATAMENTE
set_seed(42)

# ── 1. CONFIGURACION DEL EXPERIMENTO ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR_MAX = 2e-4
WEIGHT_DECAY = 1e-4

# Rutas — PROJECT_ROOT siempre relativo al propio script (robusto)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

# ── SPLITS OFICIALES SPLIT-0 (Anti Data-Leakage / Efecto Artista) ──
# NUNCA usar df.sample(). Estos TSVs garantizan que canciones del mismo
# artista NO aparecen a la vez en train y test (ver paper MTG-Jamendo).
SPLIT_DIR = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
TRAIN_TSV = SPLIT_DIR / "autotagging_moodtheme-train.tsv"
VAL_TSV   = SPLIT_DIR / "autotagging_moodtheme-validation.tsv"
TEST_TSV  = SPLIT_DIR / "autotagging_moodtheme-test.tsv"

# NOTA: benchmark=True se ELIMINA porque contradice set_seed(42).
# set_seed ya fija cudnn.deterministic=True y benchmark=False.
# Mantener benchmark=True anularía la reproducibilidad (misma razón que en train_ast.py).

def main():
    print(f"Iniciando entrenamiento SOTA de PANNs en dispositivo: {DEVICE} (bfloat16)")

    # ── 2. PREPARACION DE DATOS (Split-0 Oficial, sin Data Leakage) ──
    print("Cargando splits oficiales split-0 de MTG-Jamendo...")
    df_train = load_split_tsv(TRAIN_TSV)
    df_val   = load_split_tsv(VAL_TSV)
    # df_test  = load_split_tsv(TEST_TSV)  # Reservado para evaluación final

    print(f"  Train: {len(df_train)} canciones | Val: {len(df_val)} canciones")

    # CORRECCIÓN DEFINITIVA: las 3 etiquetas restantes del benchmark NO aparecen
    # en ninguno de los 3 splits (train/val/test) porque los tracks que las tenían
    # fueron excluidos durante la creación del split-0. La unión de los 3 splits
    # sigue dando 56. La única fuente con las 59 etiquetas es autotagging_moodtheme.tsv.
    # Derivar etiquetas de los metadatos del benchmark NO es data leakage.
    _FULL_TSV = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
    _todas: set = set()
    with open(_FULL_TSV, "r", encoding="utf-8") as _fh:
        next(_fh)  # saltar cabecera
        for _ln in _fh:
            _parts = _ln.strip().split("\t")
            if len(_parts) >= 6:
                _todas.update(
                    t.replace("mood/theme---", "").strip()
                    for t in _parts[5:]
                    if t.startswith("mood/theme---")
                )
    etiquetas_jamendo = sorted(_todas)
    print(f"  Vocabulario completo del benchmark: {len(etiquetas_jamendo)} etiquetas.")  # → 59

    preprocesador = AudioPreprocessor(target_sr=32000)
    train_ds = MTGJamendoDataset(df_train, AUDIO_DIR, preprocesador, etiquetas_jamendo, return_raw=True)
    val_ds = MTGJamendoDataset(df_val, AUDIO_DIR, preprocesador, etiquetas_jamendo, return_raw=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ── 3. MODELO Y OPTIMIZADORES ──
    model = build_panns_model(num_classes=59, device=DEVICE)
    pos_weights = train_ds.get_pos_weight().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX/10, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    # ── 4. BUCLE DE ENTRENAMIENTO ──
    history = [] 
    best_val_auc = 0.0

    print("\nComienza el entrenamiento a maxima velocidad (bfloat16)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True) 
            
            # MAGIA: Autocast con bfloat16 (Inmune a NaNs, rápido como FP16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
                
            if not torch.isfinite(loss):
                continue

            # Al usar bfloat16 no necesitamos Scaler, es estable de forma nativa
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # ── VALIDACION LIMPIA ──
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                val_loss += loss.item() if torch.isfinite(loss) else 0.0
                all_preds.append(torch.sigmoid(logits).cpu().float().numpy())
                all_targets.append(y.cpu().float().numpy())
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        y_true = np.vstack(all_targets)
        y_score = np.vstack(all_preds)
        
        valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
        if len(valid_cols) > 0:
            val_auc = roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')
        else:
            val_auc = 0.5

        print(f"Resumen Epoca {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")
        
        history.append({
            'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_auc': val_auc
        })
        # CORRECCIÓN: guardar en SCRIPT_DIR para que evaluacion_final.py los encuentre
        # independientemente del directorio desde el que se lance el script.
        pd.DataFrame(history).to_csv(SCRIPT_DIR / "history_panns.csv", index=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SCRIPT_DIR / "best_panns_model.pth")
            print(f"Guardado nuevo mejor modelo (AUC: {best_val_auc:.4f})")

    # ── GRÁFICAS DE CURVAS DE APRENDIZAJE (PNG para la memoria) ─────────────
    _save_learning_curves(
        csv_path   = SCRIPT_DIR / "history_panns.csv",
        out_path   = SCRIPT_DIR / "panns_learning_curves.png",
        model_name = "PANNs CNN14",
    )
    print(f"\nEntrenamiento PANNs finalizado. Mejor AUC validación: {best_val_auc:.4f}")


def _save_learning_curves(csv_path, out_path, model_name: str) -> None:
    """Genera y guarda las curvas de Loss y ROC-AUC como PNG.

    Idéntica a la función de train_ast.py y train_multimodal.py.
    Se puede llamar al final del entrenamiento y también desde los notebooks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[AVISO] No se pudo leer {csv_path}: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Curvas de Aprendizaje", fontsize=14, fontweight="bold")

    # ── Pérdida ──────────────────────────────────────────────────────────────
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss",
             color="#1f77b4", lw=2)
    ax1.plot(df["epoch"], df["val_loss"], label="Val Loss",
             color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("BCEWithLogitsLoss")
    ax1.set_title("Función de Pérdida", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── ROC-AUC ──────────────────────────────────────────────────────────────
    ax2.plot(df["epoch"], df["val_auc"], color="#2ca02c", lw=2,
             marker="o", markersize=4, label="Val ROC-AUC")
    best_row = df.loc[df["val_auc"].idxmax()]
    ax2.axhline(best_row["val_auc"], color="red", ls="--", alpha=0.5,
                label=f"Máx {best_row['val_auc']:.4f} (ep.{int(best_row['epoch'])})")
    ax2.axhline(0.725, color="gray", ls=":", alpha=0.7, label="Baseline VGG-ish 0.725")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("ROC-AUC Macro")
    ax2.set_title("ROC-AUC en Validación", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] Curvas de aprendizaje guardadas: {out_path}")


if __name__ == '__main__':
    main()