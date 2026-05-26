#!/usr/bin/env python3
"""
fase6b_sin_compresor.py -- Ablacion cuantitativa del compresor logaritmico
==========================================================================
TFG Reconocimiento de Emociones Musicales -- Ruben Jose Tenza Martin

OBJETIVO
--------
Reentrena PANNs sobre el mismo split-0 oficial, pero sustituyendo
AmplitudeToDB por la identidad (alimentando el modelo con la energia mel
en escala LINEAL). La diferencia en ROC-AUC Macro y PR-AUC Macro frente
al modelo canonico de la Fase 4 cuantifica el beneficio numerico de
la compresion logaritmica.

ALCANCE
-------
La ablacion se limita a PANNs. En AST el mel-espectrograma lo genera el
ASTFeatureExtractor preentrenado, no el preprocesador del proyecto, de modo
que la compresion logaritmica no es una variable de diseno libre y la
ablacion no seria metodologicamente comparable con la Fase 4.

USO
---
    cd TFG_Emociones/src
    python fase6b_sin_compresor.py --epochs 10

SALIDAS (en resultados_finales/)
--------------------------------
* fase6b_resultados.csv         -- tabla canonico vs sin_comp
* fase6b_history_panns.csv      -- curva de aprendizaje sin_comp PANNs
* fase6b_test_panns.csv         -- metricas de test sin_comp PANNs
* fase6b_panns_sin_comp.pth     -- pesos del mejor checkpoint por validacion
* fase6b_barras.png             -- comparativa visual canonico vs sin_comp
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Modulos del proyecto. Firmas reales verificadas en train_panns.py:
#   MTGJamendoDataset(df, audio_dir, preprocessor, label_list, return_raw=False)
from dataset          import MTGJamendoDataset, load_split_tsv
from audio_utils      import AudioPreprocessor
from models_panns     import build_panns_model
from evaluacion_final import compute_all_metrics

AUDIO_DIR  = PROJECT_ROOT / "data" / "audio"
SPLIT_DIR  = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
TRAIN_TSV  = SPLIT_DIR / "autotagging_moodtheme-train.tsv"
VAL_TSV    = SPLIT_DIR / "autotagging_moodtheme-validation.tsv"
TEST_TSV   = SPLIT_DIR / "autotagging_moodtheme-test.tsv"
FULL_TSV   = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"

OUT_DIR    = PROJECT_ROOT / "resultados_finales"
OUT_DIR.mkdir(exist_ok=True)

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    import os, random
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# --------------------------------------------------------------------------
# Vocabulario completo de 59 etiquetas (mismo metodo que train_panns.py)
# --------------------------------------------------------------------------
def load_full_vocab() -> list:
    """Deriva las 59 etiquetas oficiales del benchmark desde autotagging_moodtheme.tsv."""
    todas: set = set()
    with open(FULL_TSV, "r", encoding="utf-8") as fh:
        next(fh)  # cabecera
        for ln in fh:
            parts = ln.strip().split("\t")
            if len(parts) >= 6:
                todas.update(
                    t.replace("mood/theme---", "").strip()
                    for t in parts[5:] if t.startswith("mood/theme---")
                )
    return sorted(todas)  # 59 etiquetas


# --------------------------------------------------------------------------
# DONDE SE ABLACIONA EL COMPRESOR LOGARITMICO
# --------------------------------------------------------------------------
# PANNs CNN14 ingiere la forma de onda cruda (return_raw=True) y calcula su
# mel-espectrograma con un extractor INTERNO (model.spectrogram_extractor).
# La compresion logaritmica vive en su submodulo db_transform =
# AmplitudeToDB(top_db=80), NO en el AudioPreprocessor externo del proyecto.
# Por tanto la ablacion debe aplicarse sobre el propio modelo: en
# train_one_config() se sustituye ese submodulo por nn.Identity(). Como
# AmplitudeToDB no tiene parametros entrenables, la sustitucion no afecta a
# ningun peso preentrenado de los bloques convolucionales. El preprocesador
# externo se usa aqui unicamente para cargar y recortar el audio.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Helper para obtener pos_weight (corpus completo, igual que train_panns.py)
# --------------------------------------------------------------------------
def compute_pos_weight(df, label_list: list) -> torch.Tensor:
    n_total = len(df)
    n_classes = len(label_list)
    label2idx = {l: i for i, l in enumerate(label_list)}
    pos = np.zeros(n_classes, dtype=np.float64)
    for tags in df["tags"]:
        for t in tags:
            if t in label2idx:
                pos[label2idx[t]] += 1
    pos = np.maximum(pos, 1e-6)
    weight = (n_total - pos) / pos
    return torch.tensor(weight, dtype=torch.float32)


# --------------------------------------------------------------------------
# Bucle de entrenamiento (PANNs) con preproceso sin_comp
# --------------------------------------------------------------------------
def train_one_config(model_name: str, epochs: int = 10) -> dict:
    """Entrena un modelo en la variante 'sin_comp' y devuelve sus metricas."""
    set_seed(SEED)

    if model_name == "panns":
        target_sr, batch_size, lr_max, accum_steps = 32000, 16, 2e-4, 1
    else:
        raise ValueError(
            f"Fase 6B se limita a PANNs; recibido: {model_name!r}. "
            "La ablacion del compresor no aplica a AST: su mel-espectrograma "
            "lo genera el ASTFeatureExtractor preentrenado, no el preprocesador."
        )

    prep = AudioPreprocessor(target_sr=target_sr)
    vocab = load_full_vocab()
    print(f"  Vocabulario: {len(vocab)} etiquetas")

    df_train = load_split_tsv(TRAIN_TSV)
    df_val   = load_split_tsv(VAL_TSV)
    df_test  = load_split_tsv(TEST_TSV)
    print(f"  Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}")

    ds_train = MTGJamendoDataset(df_train, AUDIO_DIR, prep, vocab, return_raw=True)
    ds_val   = MTGJamendoDataset(df_val,   AUDIO_DIR, prep, vocab, return_raw=True)
    ds_test  = MTGJamendoDataset(df_test,  AUDIO_DIR, prep, vocab, return_raw=True)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    # Modelo (Fase 6B se limita a PANNs: pipeline de espectrograma controlado)
    model = build_panns_model(num_classes=len(vocab)).to(DEVICE)

    # --- ABLACION REAL DEL COMPRESOR LOGARITMICO ---
    # Se sustituye el AmplitudeToDB del extractor INTERNO de CNN14 por la
    # identidad: a partir de aqui el modelo recibe el mel-espectrograma en
    # escala LINEAL (energia), no en dB. Esta es la unica variable que
    # cambia respecto al PANNs canonico de la Fase 4.
    model.spectrogram_extractor.db_transform = nn.Identity()
    print("  [ablacion] CNN14.spectrogram_extractor.db_transform -> Identity "
          "(mel en escala LINEAL)")

    pos_weight = compute_pos_weight(df_train, vocab).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4)
    total_steps_per_epoch = math.ceil(len(dl_train) / accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max,
        steps_per_epoch=total_steps_per_epoch, epochs=epochs,
        pct_start=0.3, anneal_strategy="cos",
    )

    history = []
    best_val_auc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(dl_train):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss   = criterion(logits, y) / accum_steps
            loss.backward()
            train_losses.append(loss.item() * accum_steps)
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Validacion
        model.eval()
        val_losses, y_true, y_score = [], [], []
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss   = criterion(logits, y)
                val_losses.append(loss.item())
                y_true .append(y.float().cpu().numpy())
                y_score.append(torch.sigmoid(logits).float().cpu().numpy())
        y_true  = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        m = compute_all_metrics(y_true, y_score, model_name, vocab)

        record = {
            "epoch":      epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss":   float(np.mean(val_losses)),
            "val_auc":    float(m["ROC-AUC Macro"]),
        }
        history.append(record)
        print(f"[{model_name}|sin_comp] epoch {epoch:2d} -- "
              f"train_loss={record['train_loss']:.4f}  "
              f"val_loss={record['val_loss']:.4f}  "
              f"val_auc={record['val_auc']:.4f}")

        if m["ROC-AUC Macro"] > best_val_auc:
            best_val_auc = m["ROC-AUC Macro"]
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Test con el mejor checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in dl_test:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            y_true .append(y.float().cpu().numpy())
            y_score.append(torch.sigmoid(logits).float().cpu().numpy())
    y_true  = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    final = compute_all_metrics(y_true, y_score, model_name + " [sin_comp]", vocab)
    final["model"]        = model_name
    final["variant"]      = "sin_comp"
    final["best_val_auc"] = best_val_auc

    # Persistencia inmediata: guardamos pesos, metricas de test y curva en cuanto
    # el modelo termina, para que un fallo posterior no eche a perder el entreno.
    if best_state is not None:
        torch.save(best_state, OUT_DIR / f"fase6b_{model_name}_sin_comp.pth")
        print(f"  [PTH] fase6b_{model_name}_sin_comp.pth")
    pd.DataFrame([final]).to_csv(OUT_DIR / f"fase6b_test_{model_name}.csv", index=False)
    print(f"  [CSV] fase6b_test_{model_name}.csv")
    pd.DataFrame(history).to_csv(OUT_DIR / f"fase6b_history_{model_name}.csv", index=False)
    print(f"  [CSV] fase6b_history_{model_name}.csv")
    return final


# --------------------------------------------------------------------------
# Generacion de la tabla agregada y figura de barras
# --------------------------------------------------------------------------
def export_tabla_y_figuras(resultados_sin: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fase4 = OUT_DIR / "fase4_resultados.csv"
    if fase4.exists():
        df_fase4 = pd.read_csv(fase4)
        canon = {
            "panns": df_fase4.loc[df_fase4["Modelo"] == "PANNs CNN14"].iloc[0],
        }
    else:
        canon = None
        print(f"AVISO: {fase4} no existe; se omite la columna canonica.")

    rows = []
    for r in resultados_sin:
        if canon is not None:
            rows.append({"modelo": r["model"], "variante": "canonico (log-mel)",
                         "roc_auc_macro": canon[r["model"]]["ROC-AUC Macro"],
                         "pr_auc_macro":  canon[r["model"]]["PR-AUC Macro"]})
        rows.append({"modelo": r["model"], "variante": "sin_comp (mel lineal)",
                     "roc_auc_macro": r["ROC-AUC Macro"],
                     "pr_auc_macro":  r["PR-AUC Macro"]})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "fase6b_resultados.csv", index=False)
    print("\nResultados guardados -> fase6b_resultados.csv")
    print(df_out)

    fig, ax = plt.subplots(figsize=(8, 5))
    modelos = df_out["modelo"].unique()
    x = np.arange(len(modelos))
    width = 0.35
    if canon is not None:
        can_auc = [df_out[(df_out["modelo"] == m) & (df_out["variante"].str.startswith("canonico"))]["roc_auc_macro"].iloc[0] for m in modelos]
        ax.bar(x - width/2, can_auc, width, label="canonico (log-mel)", color="#1f4e79")
    sin_auc = [df_out[(df_out["modelo"] == m) & (df_out["variante"].str.startswith("sin_comp"))]["roc_auc_macro"].iloc[0] for m in modelos]
    ax.bar(x + width/2, sin_auc, width, label="sin compresor (lineal)", color="#a00000")
    ax.axhline(y=0.725, linestyle="--", color="gray", label="baseline VGG-ish")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in modelos])
    ax.set_ylabel("ROC-AUC Macro (test)")
    ax.set_title("Fase 6B: ablacion del compresor logaritmico")
    ax.set_ylim(0.55, 0.78); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fase6b_barras.png", dpi=130, bbox_inches="tight")
    print("Figura guardada      -> fase6b_barras.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["panns"], default="panns",
                        help="Fase 6B se limita a PANNs (unico modelo con el "
                             "pipeline de espectrograma totalmente controlado).")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"[Fase 6B] Ablacion del compresor logaritmico (solo PANNs)")
    print(f"          dispositivo = {DEVICE}")
    print(f"          modelo      = {args.model}")
    print(f"          epochs      = {args.epochs}")

    t0 = time.time()
    resultados = [train_one_config("panns", epochs=args.epochs)]

    export_tabla_y_figuras(resultados)
    print(f"\nFase 6B completada en {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
