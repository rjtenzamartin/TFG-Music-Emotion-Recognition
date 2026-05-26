"""
train_ast.py — Entrenamiento SOTA de AST (Transformer)
======================================================
TFG: Reconocimiento de Emociones Musicales

Tecnicas aplicadas: Gradient Accumulation y Bfloat16
para poder entrenar Transformers masivos en mi GPU local de forma estable.
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

# --- MIS MODULOS ---
from dataset import MTGJamendoDataset, load_split_tsv
from audio_utils import AudioPreprocessor
from models_ast import build_ast_model

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

# ── 1. CONFIGURACION EXTREMA PARA OOM (Out Of Memory) ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2             # Lote muy pequeno para que quepa en mi VRAM
ACCUMULATION_STEPS = 8     # Simulo un batch real de 16 (2 * 8)
EPOCHS = 20
LR_MAX = 5e-5              # Los Transformers necesitan un LR mas conservador que las CNNs
WEIGHT_DECAY = 1e-4

# CORRECCIÓN: Path relativo al propio script, no al directorio de ejecución.
# Path.cwd() es frágil: cambia según desde dónde lances el script.
# Path(__file__).resolve() siempre apunta al mismo sitio.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

# ── SPLITS OFICIALES SPLIT-0 (Anti Data-Leakage / Efecto Artista) ──
SPLIT_DIR = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
TRAIN_TSV = SPLIT_DIR / "autotagging_moodtheme-train.tsv"
VAL_TSV   = SPLIT_DIR / "autotagging_moodtheme-validation.tsv"
TEST_TSV  = SPLIT_DIR / "autotagging_moodtheme-test.tsv"

# NOTA: benchmark=True se ELIMINA porque contradice set_seed(42).
# set_seed ya fija cudnn.deterministic=True y benchmark=False.
# Mantener benchmark=True anularía la reproducibilidad.

def main():
    print(f"Iniciando Fine-Tuning de AST en {DEVICE} (bfloat16 puro)")

    # ── 2. PREPARACION DE DATOS (Split-0 Oficial, sin Data Leakage) ──
    print("Cargando splits oficiales split-0 de MTG-Jamendo...")
    df_train = load_split_tsv(TRAIN_TSV)
    df_val   = load_split_tsv(VAL_TSV)
    # df_test  = load_split_tsv(TEST_TSV)  # Reservado para evaluación final

    print(f"  Train: {len(df_train)} canciones | Val: {len(df_val)} canciones")

    # CORRECCIÓN DEFINITIVA: las 3 etiquetas restantes del benchmark NO aparecen
    # en ninguno de los 3 splits porque los tracks que las contenían fueron
    # excluidos durante la creación del split-0. La unión de splits solo da 56.
    # La única fuente con las 59 etiquetas es autotagging_moodtheme.tsv.
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

    # Importante: AST trabaja a 16kHz, no a 32kHz como PANNs
    preprocesador = AudioPreprocessor(target_sr=16000)

    train_ds = MTGJamendoDataset(df_train, AUDIO_DIR, preprocesador, etiquetas_jamendo, return_raw=True)
    val_ds = MTGJamendoDataset(df_val, AUDIO_DIR, preprocesador, etiquetas_jamendo, return_raw=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ── 3. MODELO Y OPTIMIZADOR ──
    model, feature_extractor = build_ast_model(num_classes=59, device=DEVICE)

    # Calculo mis pesos para contrarrestar el desbalanceo
    pos_weights = train_ds.get_pos_weight().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX/10, weight_decay=WEIGHT_DECAY)
    
    # Adapto el scheduler a mis pasos de acumulacion
    # CORRECCIÓN: usar ceil en lugar de floor para que total_steps >= pasos reales.
    # Con floor, el ultimo batch del epoch genera un step extra y OneCycleLR crashea.
    import math
    total_steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=total_steps_per_epoch, epochs=EPOCHS
    )

    # ── 4. BUCLE DE ENTRENAMIENTO CON ACUMULACION Y BFLOAT16 ──
    history = []
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for i, (x, y) in enumerate(pbar):
            y = y.to(DEVICE)
            
            # Convierto la onda cruda a los tensores que entiende el Transformer
            inputs = feature_extractor(x.numpy(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(DEVICE)
            
            # Uso bfloat16 para velocidad sin sacrificar la estabilidad numerica
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_values)
                loss = criterion(outputs.logits, y)
                # Divido el error porque estoy acumulando gradientes
                loss = loss / ACCUMULATION_STEPS 
                
            # Cortafuegos para no ensuciar la acumulacion si hay un error matematico
            if not torch.isfinite(loss):
                continue

            loss.backward()
            
            # Solo ajusto los pesos cuando he acumulado mi batch objetivo (16)
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # Cortafuegos extra: evito que gradientes rebeldes hagan explotar el Transformer
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Restauro el valor del loss visual multiplicandolo por los pasos
            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({'Loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            
        # ── VALIDACION LIMPIA ──
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                y = y.to(DEVICE)
                inputs = feature_extractor(x.numpy(), sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_values)
                    loss = criterion(outputs.logits, y)
                    
                val_loss += loss.item() if torch.isfinite(loss) else 0.0
                all_preds.append(torch.sigmoid(outputs.logits).cpu().float().numpy())
                all_targets.append(y.cpu().float().numpy())
                
        # Calculo las medias
        avg_train_loss = train_loss / len(train_loader) # Usamos la division simple para orientacion
        avg_val_loss = val_loss / len(val_loader)
        
        y_true = np.vstack(all_targets)
        y_score = np.vstack(all_preds)
        
        # Me protejo contra columnas que no tengan muestras positivas en validacion
        valid_cols = [j for j in range(y_true.shape[1]) if len(np.unique(y_true[:, j])) > 1]
        if len(valid_cols) > 0:
            val_auc = roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')
        else:
            val_auc = 0.5

        print(f"Resumen Epoca {epoch}: Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_auc': val_auc,
        })
        pd.DataFrame(history).to_csv(SCRIPT_DIR / "history_ast.csv", index=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SCRIPT_DIR / "best_ast_model.pth")
            print(f"  >> Nuevo mejor modelo guardado (AUC: {best_val_auc:.4f})")

    # ── 5. GRÁFICAS DE CURVAS DE APRENDIZAJE (PNG para la memoria) ──────────
    _save_learning_curves(
        csv_path   = SCRIPT_DIR / "history_ast.csv",
        out_path   = SCRIPT_DIR / "ast_learning_curves.png",
        model_name = "AST Transformer",
    )
    print(f"\nEntrenamiento AST finalizado. Mejor AUC validación: {best_val_auc:.4f}")


def _save_learning_curves(csv_path, out_path, model_name: str) -> None:
    """Genera y guarda las curvas de Loss y ROC-AUC como PNG.

    Se puede llamar al final del entrenamiento y también desde los notebooks.
    Requiere solo el CSV de historial — no necesita el modelo cargado.
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


if __name__ == "__main__":
    main()
