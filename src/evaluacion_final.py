#!/usr/bin/env python3
"""
evaluacion_final.py
===================
Script de evaluación final del TFG:
  "Reconocimiento de Emociones Musicales mediante Fine-Tuning y Zero-Shot"

Fases cubiertas
---------------
  FASE 4  — Métricas finales sobre test.tsv  (segmento 0-30s)
  FASE 5  — Robustez temporal (segmento 30-60s)  → RQ4
  FASE 6A — Ablación del modelo Multimodal        → RQ3
  FASE 7  — Zero-Shot CLAP vs Fine-Tuning         → RQ1

Uso
---
  python evaluacion_final.py --fase all          # todas las fases
  python evaluacion_final.py --fase 4 5          # solo FASE 4 y 5
  python evaluacion_final.py --fase 6a           # solo ablación
  python evaluacion_final.py --fase 7 tabla      # RQ1 + tabla global

Salidas
-------
  Todos los artefactos se guardan en: <PROJECT_ROOT>/resultados_finales/
  · *.npy  → arrays y_true / y_score por modelo y fase (reutilizables)
  · *.csv  → tablas de métricas (importables en la memoria del TFG)
  · *.png  → figuras publication-ready (300 dpi)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")                 # sin display (compatible con servidores)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Fija todas las semillas conocidas para reproducibilidad.

    IMPORTANTE: NO usa torch.use_deterministic_algorithms(True) porque
    crashea con los Transformers de HuggingFace (operaciones no deterministas
    en CUDA para attention con bfloat16).
    """
    os.environ["PYTHONHASHSEED"]        = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Función de inicialización para workers del DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CONFIGURACIÓN CENTRALIZADA
# ══════════════════════════════════════════════════════════════════════════════

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Resultados ─────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "resultados_finales"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Splits oficiales split-0 (Anti-Artist-Effect) ─────────────────────────
SPLIT_DIR = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
TRAIN_TSV = SPLIT_DIR / "autotagging_moodtheme-train.tsv"
TEST_TSV  = SPLIT_DIR / "autotagging_moodtheme-test.tsv"

# ── Datos ──────────────────────────────────────────────────────────────────
AUDIO_DIR      = PROJECT_ROOT / "data" / "audio"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings_fase3"

# ── Modelos ← AJUSTA ESTAS RUTAS CUANDO TERMINE EL REENTRENAMIENTO ─────────
PANNS_MODEL_PATH      = SCRIPT_DIR / "best_panns_model.pth"
AST_MODEL_PATH        = SCRIPT_DIR / "best_ast_model.pth"
MULTIMODAL_MODEL_PATH = SCRIPT_DIR / "best_multimodal_model.pth"

# ── Zero-Shot CLAP (resultados pre-calculados del notebook de zero-shot) ────
CLAP_Y_TRUE_PATH  = PROJECT_ROOT / "data" / "embeddings" / "zero_shot_y_true_global.npy"
CLAP_Y_SCORE_PATH = PROJECT_ROOT / "data" / "embeddings" / "zero_shot_y_score_global.npy"

# ── Hiperparámetros de inferencia ──────────────────────────────────────────
BATCH_PANNS  = 16
BATCH_AST    = 2    # AST es muy grande; batch pequeño para no OOM
BATCH_MULTI  = 64
NUM_WORKERS  = 8

# ── Referencia del baseline oficial MediaEval 2021 ────────────────────────
BASELINE_ROC_AUC = 0.725   # VGG-ish, Bogdanov et al. 2019


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — UTILIDADES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════════════

def load_split_tsv(tsv_path: Path) -> pd.DataFrame:
    """Carga un TSV oficial de MTG-Jamendo (split-0) de forma robusta."""
    registros = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f)                          # saltar cabecera
        for linea in f:
            if not linea.strip():
                continue
            parts = linea.strip().split("\t")
            if len(parts) >= 6:
                tags = [
                    t.replace("mood/theme---", "").strip()
                    for t in parts[5:]
                    if t.startswith("mood/theme---")
                ]
                if tags:
                    registros.append({"path": parts[3], "tags": tags})
    return pd.DataFrame(registros)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    etiquetas: list[str],
) -> dict:
    """
    Calcula el conjunto COMPLETO de métricas multilabel necesarias para el TFG.

    Métricas incluidas y su justificación en las RQs
    -------------------------------------------------
    ROC-AUC Macro/Micro   → métrica principal del baseline VGG-ish y MediaEval
    PR-AUC Macro (mAP)    → métrica oficial MediaEval 2021; más informativa
                            que ROC-AUC con clases desbalanceadas
    LRAP                  → Label Ranking Average Precision; evalúa la calidad
                            del ranking sin depender de umbral
    F1 Macro (umbral opt) → exigido por el profesor; umbral óptimo por clase
                            maximiza F1 individualmente (≠ umbral global 0.5)
    F1 Samples (0.5)      → perspectiva por-muestra con umbral fijo
    Hamming Loss          → fracción de etiquetas incorrectas; métrica estándar
                            de clasificación multilabel
    Spearman ρ(soporte↔AUC) → ¿limita el desbalanceo el rendimiento? (RQ2/RQ3)

    Parameters
    ----------
    y_true      : (N, 59) ground truth binario
    y_score     : (N, 59) probabilidades predichas (post-sigmoid)
    model_name  : nombre del modelo para el campo 'Modelo' del resultado
    etiquetas   : lista de etiquetas (para Spearman y nombre)
    """
    # Clases con al menos un positivo Y un negativo en este split
    valid_cols = [
        i for i in range(y_true.shape[1])
        if len(np.unique(y_true[:, i])) > 1
    ]
    yt = y_true[:, valid_cols]
    ys = y_score[:, valid_cols]

    # ── Métricas independientes del umbral ──────────────────────────────────
    roc_macro = roc_auc_score(yt, ys, average="macro")
    roc_micro = roc_auc_score(yt, ys, average="micro")
    pr_macro  = average_precision_score(yt, ys, average="macro")
    pr_micro  = average_precision_score(yt, ys, average="micro")
    lrap      = label_ranking_average_precision_score(yt, ys)

    # ── F1 con umbral óptimo por clase ──────────────────────────────────────
    # Justificación matemática: con distribuciones de clase muy distintas,
    # un umbral global de 0.5 es subóptimo. Optimizar por clase es el
    # protocolo estándar en los benchmarks de multilabel audio tagging.
    print(f"  [{model_name}] Optimizando umbrales F1 por clase...")
    best_thresholds = np.full(y_true.shape[1], 0.5)
    for c in valid_cols:
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.05, 0.95, 0.05):
            score = f1_score(
                y_true[:, c],
                (y_score[:, c] >= th).astype(int),
                zero_division=0,
            )
            if score > best_f1:
                best_f1, best_th = score, th
        best_thresholds[c] = best_th

    y_pred_opt    = (y_score >= best_thresholds).astype(int)
    f1_macro_opt  = f1_score(y_true, y_pred_opt, average="macro",  zero_division=0)
    f1_micro_opt  = f1_score(y_true, y_pred_opt, average="micro",  zero_division=0)

    # ── F1 con umbral fijo 0.5 (perspectiva por muestra) ────────────────────
    y_pred_05   = (y_score >= 0.5).astype(int)
    f1_samples  = f1_score(y_true, y_pred_05, average="samples", zero_division=0)
    f1_macro_05 = f1_score(y_true, y_pred_05, average="macro",   zero_division=0)
    hl          = hamming_loss(y_true, y_pred_05)

    # ── Correlación de Spearman: soporte vs AUC ─────────────────────────────
    # Si ρ > 0 y p < 0.05 → clases con más ejemplos se predicen mejor.
    # Esto confirma que el desbalanceo sigue siendo el factor limitante (RQ).
    soporte   = y_true.sum(axis=0)[valid_cols]
    auc_por_c = [roc_auc_score(y_true[:, i], y_score[:, i]) for i in valid_cols]
    spearman_r, spearman_p = spearmanr(soporte, auc_por_c)

    return {
        "Modelo"           : model_name,
        "ROC-AUC Macro"    : roc_macro,
        "ROC-AUC Micro"    : roc_micro,
        "PR-AUC Macro"     : pr_macro,
        "PR-AUC Micro"     : pr_micro,
        "LRAP"             : lrap,
        "F1 Macro (opt)"   : f1_macro_opt,
        "F1 Micro (opt)"   : f1_micro_opt,
        "F1 Macro (0.5)"   : f1_macro_05,
        "F1 Samples (0.5)" : f1_samples,
        "Hamming Loss"     : hl,
        "Spearman rho"     : spearman_r,
        "Spearman p"       : spearman_p,
        "n_valid_clases"   : len(valid_cols),
        "Delta_Baseline"   : roc_macro - BASELINE_ROC_AUC,
    }


def print_results(r: dict) -> None:
    """Imprime una tabla ASCII con los resultados de un modelo."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  {r['Modelo']}")
    print(sep)
    print(f"  ROC-AUC  Macro : {r['ROC-AUC Macro']:.4f}  │  Micro: {r['ROC-AUC Micro']:.4f}")
    print(f"  PR-AUC   Macro : {r['PR-AUC Macro']:.4f}  │  Micro: {r['PR-AUC Micro']:.4f}")
    print(f"  LRAP           : {r['LRAP']:.4f}")
    print(f"  F1 Macro (opt) : {r['F1 Macro (opt)']:.4f}  │  Micro: {r['F1 Micro (opt)']:.4f}")
    print(f"  F1 Macro (0.5) : {r['F1 Macro (0.5)']:.4f}  │  Samples: {r['F1 Samples (0.5)']:.4f}")
    print(f"  Hamming Loss   : {r['Hamming Loss']:.4f}")
    print(f"  Spearman ρ     : {r['Spearman rho']:.3f}  (p={r['Spearman p']:.4f})")
    print(f"  Δ vs Baseline  : {r['Delta_Baseline']:+.4f}  (baseline VGG-ish = {BASELINE_ROC_AUC})")
    print(f"  Clases válidas : {r['n_valid_clases']}/59")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — DATASET GENÉRICO CON SOPORTE DE OFFSET TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

class AudioWindowDataset(Dataset):
    """
    Dataset de inferencia que carga una ventana de 30s con offset configurable.

    El parámetro ``start_sec`` permite cambiar de segmento con una sola línea:
      · start_sec=0.0  → FASE 4 (segmento 0-30s, igual que entrenamiento)
      · start_sec=30.0 → FASE 5 (segmento 30-60s, experimento RQ4)

    Garantías de determinismo (importantes para RQ4):
    - NO se aplica librosa.effects.trim: el offset es siempre exacto en muestras.
    - Si el audio no llega a start_sec + 30s, se rellena con ceros (zero-pad),
      NO con silencio detectado → comportamiento idéntico para todas las canciones.
    - librosa.load(offset=start_sec, duration=30.0) opera directamente sobre el
      archivo de audio original sin modificar el origen temporal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        target_sr: int,
        etiquetas: list[str],
        start_sec: float = 0.0,
    ) -> None:
        self.audio_dir   = Path(audio_dir)
        self.target_sr   = target_sr
        self.etiquetas   = etiquetas
        self.etiq_idx    = {e: i for i, e in enumerate(etiquetas)}
        self.start_sec   = start_sec
        self.target_samp = int(target_sr * 30)

        # Los audios están en formato .low.mp3 (versión comprimida del dataset)
        # El TSV oficial referencia rutas con .mp3 → reemplazamos la extensión.
        df = df.reset_index(drop=True)
        df["path"] = df["path"].str.replace(".mp3", ".low.mp3", regex=False)
        mask = df["path"].apply(lambda p: (self.audio_dir / p).exists())
        n_missing = int((~mask).sum())
        if n_missing > 0:
            print(f"  [AVISO] {n_missing}/{len(df)} tracks sin audio en disco — excluidos.")
        self.df = df[mask].reset_index(drop=True)
        print(f"  Tracks evaluables: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        audio_path = self.audio_dir / row["path"]

        # Carga determinista: offset fijo, sin ningún preprocesado de silencio.
        # Try/except: algunos .low.mp3 son más cortos que start_sec segundos y
        # soundfile lanza psf_fseek() failed al intentar el seek. En ese caso
        # devolvemos silencio (ceros), que el modelo predecirá como "sin emoción".
        try:
            y, _ = librosa.load(
                str(audio_path),
                sr=self.target_sr,
                mono=True,
                offset=self.start_sec,
                duration=30.0,
            )
        except Exception:
            y = np.zeros(self.target_samp, dtype=np.float32)

        # Zero-pad si el audio es más corto que start_sec + 30s
        if len(y) < self.target_samp:
            y = np.pad(y, (0, self.target_samp - len(y)), mode="constant")
        else:
            y = y[: self.target_samp]

        # Target multilabel (one-hot sobre las 59 clases del vocabulario)
        target = np.zeros(len(self.etiquetas), dtype=np.float32)
        for tag in row["tags"]:
            if tag in self.etiq_idx:
                target[self.etiq_idx[tag]] = 1.0

        return torch.from_numpy(y), torch.from_numpy(target)


def _make_loader(ds: Dataset, batch_size: int, num_workers: int = NUM_WORKERS) -> DataLoader:
    """Helper: DataLoader con semilla fija en workers."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — FUNCIONES DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def infer_panns(model: nn.Module, loader: DataLoader, desc: str = "PANNs"):
    """Inferencia para CNN14 (acepta audio crudo, mel front-end interno)."""
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            all_preds.append(torch.sigmoid(logits).cpu().float().numpy())
            all_targets.append(y.numpy())
    return np.vstack(all_targets), np.vstack(all_preds)


def infer_ast(model: nn.Module, feature_extractor, loader: DataLoader, desc: str = "AST"):
    """Inferencia para AST (requiere feature_extractor de HuggingFace)."""
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x_raw, y in tqdm(loader, desc=desc, leave=False):
            # x_raw: (B, 480000) — audio crudo 16kHz 30s
            inputs = feature_extractor(
                x_raw.numpy(), sampling_rate=16000, return_tensors="pt"
            )
            input_values = inputs.input_values.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_values)
            probs = torch.sigmoid(outputs.logits).cpu().float().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())
    return np.vstack(all_targets), np.vstack(all_preds)


def infer_multimodal(
    model: nn.Module,
    loader: DataLoader,
    desc: str = "Multimodal",
    zero_ast: bool = False,
    zero_panns: bool = False,
    zero_whisper: bool = False,
):
    """
    Inferencia para el modelo Late Fusion Multimodal.

    Los flags ``zero_*`` permiten anular ramas individuales (FASE 6A):
      · zero_ast=True     → sustituir embedding AST por vector 0
      · zero_panns=True   → sustituir embedding PANNs por vector 0
      · zero_whisper=True → sustituir embedding Whisper por vector 0

    Justificación de la ablación por ceros (vs. ablación por ruido):
    Usar el vector nulo es la forma más limpia de desactivar una rama sin
    modificar los pesos del MLP. La capa de concatenación simplemente recibe
    una contribución nula de esa modalidad; el resto del pipeline es idéntico.
    """
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for ast_x, panns_x, whisper_x, y in tqdm(loader, desc=desc, leave=False):
            if zero_ast:     ast_x     = torch.zeros_like(ast_x)
            if zero_panns:   panns_x   = torch.zeros_like(panns_x)
            if zero_whisper: whisper_x = torch.zeros_like(whisper_x)
            ast_x, panns_x, whisper_x = (
                ast_x.to(DEVICE),
                panns_x.to(DEVICE),
                whisper_x.to(DEVICE),
            )
            logits = model(ast_x, panns_x, whisper_x)
            probs  = torch.sigmoid(logits).cpu().float().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())
    return np.vstack(all_targets), np.vstack(all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — FASE 4: MÉTRICAS FINALES EN TEST (0-30s)
# Responde a: RQ1 (paradigmas), RQ2 (arquitecturas), RQ3 (multimodalidad)
# ══════════════════════════════════════════════════════════════════════════════

def fase4_metricas_finales(etiquetas: list[str], df_test: pd.DataFrame) -> list[dict]:
    """
    Evalúa todos los modelos disponibles sobre test.tsv con el segmento 0-30s.
    Esta es la evaluación canónica del TFG — equivalente protocolo MediaEval.
    """
    print("\n" + "▓" * 62)
    print("  FASE 4 — MÉTRICAS FINALES EN TEST (0-30s)")
    print("▓" * 62)

    resultados = []

    # ── PANNs CNN14 ──────────────────────────────────────────────────────────
    if PANNS_MODEL_PATH.exists():
        print("\n[FASE 4] Cargando PANNs CNN14...")
        from models_panns import build_panns_model
        model = build_panns_model(num_classes=59, device=DEVICE)
        model.load_state_dict(torch.load(PANNS_MODEL_PATH, map_location=DEVICE))

        ds     = AudioWindowDataset(df_test, AUDIO_DIR, 32000, etiquetas, start_sec=0.0)
        loader = _make_loader(ds, BATCH_PANNS)
        y_true, y_score = infer_panns(model, loader, "PANNs [TEST 0-30s]")

        np.save(RESULTS_DIR / "panns_test_y_true.npy",  y_true)
        np.save(RESULTS_DIR / "panns_test_y_score.npy", y_score)
        r = compute_all_metrics(y_true, y_score, "PANNs CNN14", etiquetas)
        print_results(r); resultados.append(r)
        # Liberar VRAM explícitamente antes del siguiente modelo
        del model, ds, loader, y_true, y_score
        torch.cuda.empty_cache()
    else:
        print(f"\n[SKIP] PANNs — modelo no encontrado: {PANNS_MODEL_PATH}")

    # ── AST Transformer ──────────────────────────────────────────────────────
    if AST_MODEL_PATH.exists():
        print("\n[FASE 4] Cargando AST Transformer...")
        from transformers import ASTForAudioClassification, ASTFeatureExtractor
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        fe       = ASTFeatureExtractor.from_pretrained(model_id)
        model    = ASTForAudioClassification.from_pretrained(
            model_id, num_labels=59, ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(AST_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)

        ds     = AudioWindowDataset(df_test, AUDIO_DIR, 16000, etiquetas, start_sec=0.0)
        loader = _make_loader(ds, BATCH_AST)
        y_true, y_score = infer_ast(model, fe, loader, "AST [TEST 0-30s]")

        np.save(RESULTS_DIR / "ast_test_y_true.npy",  y_true)
        np.save(RESULTS_DIR / "ast_test_y_score.npy", y_score)
        r = compute_all_metrics(y_true, y_score, "AST Transformer", etiquetas)
        print_results(r); resultados.append(r)
        # Liberar VRAM explícitamente antes del siguiente modelo
        del model, fe, ds, loader, y_true, y_score
        torch.cuda.empty_cache()
    else:
        print(f"\n[SKIP] AST — modelo no encontrado: {AST_MODEL_PATH}")

    # ── Late Fusion Multimodal ────────────────────────────────────────────────
    if MULTIMODAL_MODEL_PATH.exists():
        print("\n[FASE 4] Cargando Late Fusion Multimodal...")
        from train_multimodal import MultimodalLateFusion, MultimodalJamendoDataset
        model = MultimodalLateFusion(num_classes=59).to(DEVICE)
        model.load_state_dict(torch.load(MULTIMODAL_MODEL_PATH, map_location=DEVICE))

        g = torch.Generator(); g.manual_seed(SEED)
        test_ds = MultimodalJamendoDataset(TEST_TSV, EMBEDDINGS_DIR, label_list=etiquetas)
        loader  = DataLoader(test_ds, batch_size=BATCH_MULTI, shuffle=False,
                             num_workers=4, generator=g, worker_init_fn=seed_worker)
        y_true, y_score = infer_multimodal(model, loader, "Multimodal [TEST 0-30s]")

        np.save(RESULTS_DIR / "multi_test_y_true.npy",  y_true)
        np.save(RESULTS_DIR / "multi_test_y_score.npy", y_score)
        r = compute_all_metrics(y_true, y_score, "Late Fusion Multimodal", etiquetas)
        print_results(r); resultados.append(r)
    else:
        print(f"\n[SKIP] Multimodal — modelo no encontrado: {MULTIMODAL_MODEL_PATH}")

    if resultados:
        pd.DataFrame(resultados).to_csv(RESULTS_DIR / "fase4_resultados.csv", index=False)
        print(f"\n[OK] FASE 4 → {RESULTS_DIR / 'fase4_resultados.csv'}")
        _plot_fase4_comparison(resultados)

    return resultados


def _plot_fase4_comparison(resultados: list[dict]) -> None:
    """
    Genera la figura comparativa principal de la Sección de Resultados del TFG.

    Muestra ROC-AUC, PR-AUC y F1 (umbral óptimo) de todos los modelos evaluados
    sobre test.tsv, con la línea de referencia del baseline VGG-ish.
    Esta es la figura que va en la página de resultados de la memoria.
    """
    if not resultados:
        return

    df = pd.DataFrame(resultados)
    metricas = ["ROC-AUC Macro", "PR-AUC Macro", "F1 Macro (opt)"]
    colores  = ["#2980b9", "#e67e22", "#27ae60"]

    # Añadir baseline como fila de referencia
    baseline_row = pd.DataFrame([{
        "Modelo": "VGG-ish Baseline",
        "ROC-AUC Macro": BASELINE_ROC_AUC,
        "PR-AUC Macro": None,
        "F1 Macro (opt)": None,
    }])
    df_plot = pd.concat([baseline_row, df[["Modelo"] + metricas]], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(df_plot) * 0.9)))
    fig.suptitle("FASE 4 — Métricas Finales sobre test.tsv  (segmento 0-30s)",
                 fontsize=13, fontweight="bold")

    for ax, met, col in zip(axes, metricas, colores):
        vals   = df_plot[met].astype(float, errors="ignore")
        labels = df_plot["Modelo"]
        valid  = vals.notna()

        bars = ax.barh(
            labels[valid], vals[valid],
            color=[col if "Baseline" not in str(n) else "#bdc3c7"
                   for n in labels[valid]],
            alpha=0.85, edgecolor="white", height=0.6,
        )
        # Línea de baseline ROC-AUC solo en el primer panel
        if met == "ROC-AUC Macro":
            ax.axvline(BASELINE_ROC_AUC, color="gray", ls="--", alpha=0.7,
                       label=f"Baseline {BASELINE_ROC_AUC}")
            ax.legend(fontsize=8)

        ax.set_title(met, fontweight="bold", fontsize=11)
        ax.set_xlim(0, 1)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "fase4_test_comparativa.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico FASE 4 → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — FASE 5: ROBUSTEZ TEMPORAL (30-60s)  → RQ4
# ══════════════════════════════════════════════════════════════════════════════

def fase5_robustez_temporal(etiquetas: list[str], df_test: pd.DataFrame) -> list[dict]:
    """
    Evalúa PANNs y AST con el segmento 30-60s del audio del test set.

    Protocolo experimental (para la memoria, Sección Metodología):
    ──────────────────────────────────────────────────────────────────
    Los modelos se entrenaron EXCLUSIVAMENTE con los segundos 0-30 de
    cada canción. Evaluar sobre 30-60s (que suele contener el estribillo
    o el clímax musical) expone si el modelo ha memorizado características
    del intro/segmento inicial o si ha aprendido invariantes acústicos
    independientes de la posición temporal.

    Hipótesis nula (H₀):  ROC-AUC(30-60s) ≈ ROC-AUC(0-30s)
      → el modelo generaliza por contenido acústico, no por posición.
    Hipótesis alternativa (H₁):  ROC-AUC(30-60s) << ROC-AUC(0-30s)
      → hay sesgo posicional ("Positional Bias" o memorización estructural).

    Nota sobre el modelo Multimodal:
    El Late Fusion usa embeddings pre-computados del segmento 0-30s. Re-
    extraer los embeddings para 30-60s requeriría ~72h de inferencia con
    AST+PANNs+Whisper, lo que queda fuera del alcance de este TFG. La
    evaluación de robustez temporal se realiza, por tanto, sobre los
    modelos unimodales que operan directamente sobre la señal de audio.
    """
    print("\n" + "▓" * 62)
    print("  FASE 5 — ROBUSTEZ TEMPORAL (30-60s)  →  RQ4")
    print("▓" * 62)
    print("  Garantía: sin librosa.effects.trim — offset=30.0s estricto")

    resultados = []

    # ── PANNs en 30-60s ──────────────────────────────────────────────────────
    if PANNS_MODEL_PATH.exists():
        print("\n[FASE 5] PANNs CNN14 → segmento 30-60s")
        from models_panns import build_panns_model
        model = build_panns_model(num_classes=59, device=DEVICE)
        model.load_state_dict(torch.load(PANNS_MODEL_PATH, map_location=DEVICE))

        ds     = AudioWindowDataset(df_test, AUDIO_DIR, 32000, etiquetas, start_sec=30.0)
        loader = _make_loader(ds, BATCH_PANNS)
        y_true, y_score = infer_panns(model, loader, "PANNs [TEST 30-60s]")

        np.save(RESULTS_DIR / "panns_temporal_y_true.npy",  y_true)
        np.save(RESULTS_DIR / "panns_temporal_y_score.npy", y_score)
        r = compute_all_metrics(y_true, y_score, "PANNs CNN14 [30-60s]", etiquetas)
        print_results(r); resultados.append(r)
        del model, ds, loader, y_true, y_score
        torch.cuda.empty_cache()

    # ── AST en 30-60s ────────────────────────────────────────────────────────
    if AST_MODEL_PATH.exists():
        print("\n[FASE 5] AST Transformer → segmento 30-60s")
        from transformers import ASTForAudioClassification, ASTFeatureExtractor
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        fe       = ASTFeatureExtractor.from_pretrained(model_id)
        model    = ASTForAudioClassification.from_pretrained(
            model_id, num_labels=59, ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(AST_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)

        ds     = AudioWindowDataset(df_test, AUDIO_DIR, 16000, etiquetas, start_sec=30.0)
        loader = _make_loader(ds, BATCH_AST)
        y_true, y_score = infer_ast(model, fe, loader, "AST [TEST 30-60s]")

        np.save(RESULTS_DIR / "ast_temporal_y_true.npy",  y_true)
        np.save(RESULTS_DIR / "ast_temporal_y_score.npy", y_score)
        r = compute_all_metrics(y_true, y_score, "AST Transformer [30-60s]", etiquetas)
        print_results(r); resultados.append(r)

    if resultados:
        pd.DataFrame(resultados).to_csv(RESULTS_DIR / "fase5_resultados.csv", index=False)
        print(f"\n[OK] FASE 5 → {RESULTS_DIR / 'fase5_resultados.csv'}")

    return resultados


def plot_temporal_drop() -> None:
    """
    Figura central de RQ4: caída de ROC-AUC macro entre 0-30s y 30-60s.
    Requiere que FASE 4 y FASE 5 hayan guardado sus CSV.
    """
    f4_path = RESULTS_DIR / "fase4_resultados.csv"
    f5_path = RESULTS_DIR / "fase5_resultados.csv"
    if not f4_path.exists() or not f5_path.exists():
        print("[SKIP] plot_temporal_drop: faltan CSV de FASE 4 o FASE 5.")
        return

    f4 = pd.read_csv(f4_path).set_index("Modelo")
    f5 = pd.read_csv(f5_path).set_index("Modelo")

    modelos = [m for m in ["PANNs CNN14", "AST Transformer"]
               if m in f4.index and f"{m} [30-60s]" in f5.index]
    if not modelos:
        print("[SKIP] No hay datos suficientes para el gráfico de caída temporal.")
        return

    seg0   = [f4.loc[m, "ROC-AUC Macro"] for m in modelos]
    seg30  = [f5.loc[f"{m} [30-60s]", "ROC-AUC Macro"] for m in modelos]
    drops  = [a - b for a, b in zip(seg0, seg30)]
    x      = np.arange(len(modelos))
    w      = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Barras agrupadas: 0-30s vs 30-60s
    bars1 = ax1.bar(x - w / 2, seg0,  w, label="0-30s (entrenamiento)", color="#3498db", alpha=0.87)
    bars2 = ax1.bar(x + w / 2, seg30, w, label="30-60s (evaluación)",   color="#e74c3c", alpha=0.87)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modelos, fontsize=11)
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(BASELINE_ROC_AUC, color="gray", ls="--", alpha=0.6,
                label=f"Baseline VGG-ish {BASELINE_ROC_AUC}")
    ax1.set_ylabel("ROC-AUC Macro")
    ax1.set_title("RQ4: Rendimiento por segmento temporal", fontweight="bold")
    ax1.legend(fontsize=9)
    for bar in [*bars1, *bars2]:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{bar.get_height():.3f}", ha="center", fontsize=9)

    # Caída absoluta
    colors = ["#27ae60" if d < 0.02 else "#e74c3c" for d in drops]
    ax2.bar(modelos, drops, color=colors, alpha=0.87, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_ylabel("Δ ROC-AUC  (0-30s → 30-60s)")
    ax2.set_title("Caída de rendimiento (Positional Bias)", fontweight="bold")
    for i, d in enumerate(drops):
        ax2.text(i, d + 0.001 if d >= 0 else d - 0.003,
                 f"{d:+.4f}", ha="center", fontsize=11, fontweight="bold")

    plt.suptitle("FASE 5 — Robustez Temporal  (RQ4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "fase5_temporal_drop.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[OK] Gráfico RQ4 guardado: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — FASE 6A: ABLACIÓN MULTIMODAL  → RQ3
# ══════════════════════════════════════════════════════════════════════════════

def fase6a_ablacion_multimodal(etiquetas: list[str]) -> list[dict]:
    """
    Evalúa el modelo Late Fusion anulando combinaciones de sus 3 ramas.

    Diseño del experimento (para la memoria):
    ─────────────────────────────────────────────────────────────────
    Se evalúan los 7 subsets no vacíos de {AST, PANNs, Whisper}.
    Para cada combinación activa, las ramas inactivas reciben un vector
    de ceros (dimensión original preservada). Esto garantiza que los
    pesos del MLP NO se modifican y la comparación es justa.

    Hipótesis a responder (RQ3):
      · Si Full > cada combinación individual → hay complementariedad real.
      · Si "Solo AST" ≈ Full → PANNs y Whisper no aportan información nueva.
      · La importancia relativa de Whisper (semántico) vs PANNs (acústico)
        indica si las letras son relevantes para la tarea.

    Restricción: este experimento evalúa sobre embeddings 0-30s. Para un
    análisis de ablación más profundo, se necesitarían re-extraer embeddings
    de múltiples segmentos (fuera del alcance de este TFG).
    """
    if not MULTIMODAL_MODEL_PATH.exists():
        print(f"\n[SKIP] FASE 6A — modelo no encontrado: {MULTIMODAL_MODEL_PATH}")
        return []

    print("\n" + "▓" * 62)
    print("  FASE 6A — ABLACIÓN MULTIMODAL  →  RQ3")
    print("▓" * 62)

    from train_multimodal import MultimodalLateFusion, MultimodalJamendoDataset

    model = MultimodalLateFusion(num_classes=59).to(DEVICE)
    model.load_state_dict(torch.load(MULTIMODAL_MODEL_PATH, map_location=DEVICE))

    g = torch.Generator(); g.manual_seed(SEED)
    test_ds = MultimodalJamendoDataset(TEST_TSV, EMBEDDINGS_DIR, label_list=etiquetas)
    loader  = DataLoader(test_ds, batch_size=BATCH_MULTI, shuffle=False,
                         num_workers=4, generator=g, worker_init_fn=seed_worker)

    # (zero_ast, zero_panns, zero_whisper, nombre_descriptivo)
    # El orden va de más completo a menos completo para lectura natural
    COMBINACIONES = [
        (False, False, False, "AST + PANNs + Whisper  ★ Full"),
        (False, False, True,  "AST + PANNs"),
        (False, True,  False, "AST + Whisper"),
        (True,  False, False, "PANNs + Whisper"),
        (False, True,  True,  "Solo AST"),
        (True,  False, True,  "Solo PANNs"),
        (True,  True,  False, "Solo Whisper"),
    ]

    resultados = []
    for z_ast, z_panns, z_whisper, nombre in COMBINACIONES:
        print(f"\n  ▸ {nombre}")
        y_true, y_score = infer_multimodal(
            model, loader,
            desc=f"  {nombre[:30]}",
            zero_ast=z_ast, zero_panns=z_panns, zero_whisper=z_whisper,
        )
        r = compute_all_metrics(y_true, y_score, nombre, etiquetas)
        print(f"    ROC-AUC Macro: {r['ROC-AUC Macro']:.4f}  │  "
              f"PR-AUC: {r['PR-AUC Macro']:.4f}  │  F1: {r['F1 Macro (opt)']:.4f}")
        resultados.append(r)

    df_abl = pd.DataFrame(resultados)
    df_abl.to_csv(RESULTS_DIR / "fase6a_ablacion.csv", index=False)

    # ── Figura de ablación ────────────────────────────────────────────────────
    df_plot = df_abl[["Modelo", "ROC-AUC Macro", "PR-AUC Macro", "F1 Macro (opt)"]
                     ].sort_values("ROC-AUC Macro", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metricas  = ["ROC-AUC Macro", "PR-AUC Macro", "F1 Macro (opt)"]
    colores   = ["#3498db", "#e67e22", "#2ecc71"]

    for ax, met, col in zip(axes, metricas, colores):
        bars = ax.barh(df_plot["Modelo"], df_plot[met],
                       color=col, alpha=0.82, edgecolor="white")
        # Resaltar la fila Full
        for bar, name in zip(bars, df_plot["Modelo"]):
            if "Full" in name:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)
        ax.set_title(met, fontweight="bold")
        ax.set_xlim(0, 1)
        for bar in bars:
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.3f}", va="center", fontsize=9)

    plt.suptitle("FASE 6A — Ablación Multimodal  (RQ3): Contribución por Modalidad",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "fase6a_ablacion.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[OK] FASE 6A → {RESULTS_DIR / 'fase6a_ablacion.csv'}")
    print(f"[OK] Figura   → {out}")
    plt.close()

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — FASE 7: ZERO-SHOT CLAP vs FINE-TUNING  → RQ1
# ══════════════════════════════════════════════════════════════════════════════

def fase7_zero_shot(etiquetas: list[str], df_test: pd.DataFrame):
    """
    Carga las predicciones pre-calculadas de CLAP y las compara con los
    modelos fine-tuned sobre el mismo conjunto de test.

    Protocolo de alineación:
    ─────────────────────────────────────────────────────────────────
    Los arrays .npy de CLAP pueden cubrir más canciones (train+val+test)
    dependiendo de cómo se extrajeron. El script intenta:
      1. Usar los arrays tal cual si tienen exactamente N_test filas.
      2. Tomar las últimas N_test filas (heurística válida si se extrajeron
         en el orden train → val → test).
      3. Si nada cuadra, muestra un warning descriptivo para el usuario.
    """
    print("\n" + "▓" * 62)
    print("  FASE 7 — ZERO-SHOT CLAP vs FINE-TUNING  →  RQ1")
    print("▓" * 62)

    if not CLAP_Y_TRUE_PATH.exists() or not CLAP_Y_SCORE_PATH.exists():
        print(f"\n[SKIP] Predicciones CLAP no encontradas.")
        print(f"  Ruta esperada y_true : {CLAP_Y_TRUE_PATH}")
        print(f"  Ruta esperada y_score: {CLAP_Y_SCORE_PATH}")
        print("  → Ejecuta primero el notebook de zero-shot (CLAP) y asegúrate")
        print("    de que guarda los arrays en esas rutas con np.save().")
        return None

    y_true_clap  = np.load(CLAP_Y_TRUE_PATH)
    y_score_clap = np.load(CLAP_Y_SCORE_PATH)
    n_test = len(df_test)

    print(f"  CLAP arrays cargados: {y_true_clap.shape[0]} muestras")
    print(f"  Canciones en test.tsv: {n_test}")

    if y_true_clap.shape[0] == n_test:
        print("  Alineación perfecta. Usando arrays tal cual.")
    elif y_true_clap.shape[0] > n_test:
        print(f"  Tomando las últimas {n_test} filas (asumiendo orden train→val→test).")
        y_true_clap  = y_true_clap[-n_test:]
        y_score_clap = y_score_clap[-n_test:]
    else:
        print(f"  [AVISO] El array CLAP tiene menos filas ({y_true_clap.shape[0]}) "
              f"que test.tsv ({n_test}). Verifica la extracción de embeddings.")

    r_clap = compute_all_metrics(y_true_clap, y_score_clap, "CLAP (Zero-Shot)", etiquetas)
    print_results(r_clap)

    # ── Tabla comparativa final RQ1 ───────────────────────────────────────────
    rows = [{
        "Modelo"        : "VGG-ish (Baseline MediaEval 2021)",
        "Paradigma"     : "Baseline Supervisado",
        "ROC-AUC Macro" : BASELINE_ROC_AUC,
        "PR-AUC Macro"  : None,
        "F1 Macro (opt)": None,
        "LRAP"          : None,
        "Delta_Baseline": 0.0,
    }]
    rows.append({**r_clap, "Paradigma": "Zero-Shot"})

    for prefix, nombre in [
        ("panns", "PANNs CNN14"),
        ("ast",   "AST Transformer"),
        ("multi", "Late Fusion Multimodal"),
    ]:
        t = RESULTS_DIR / f"{prefix}_test_y_true.npy"
        s = RESULTS_DIR / f"{prefix}_test_y_score.npy"
        if t.exists() and s.exists():
            r = compute_all_metrics(np.load(t), np.load(s), nombre, etiquetas)
            rows.append({**r, "Paradigma": "Fine-Tuning Supervisado"})
        else:
            rows.append({
                "Modelo": nombre, "Paradigma": "Fine-Tuning Supervisado",
                "ROC-AUC Macro": "(pendiente — ejecuta FASE 4 primero)",
            })

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(RESULTS_DIR / "fase7_rq1_comparativa.csv", index=False)

    # ── Figura RQ1 ────────────────────────────────────────────────────────────
    df_plot = df_cmp[
        df_cmp["ROC-AUC Macro"].apply(lambda x: isinstance(x, float))
    ][["Modelo", "ROC-AUC Macro", "Paradigma"]].copy()

    palette = {
        "Baseline Supervisado"     : "#95a5a6",
        "Zero-Shot"                 : "#e74c3c",
        "Fine-Tuning Supervisado"  : "#2980b9",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, row) in enumerate(df_plot.iterrows()):
        color = palette.get(row["Paradigma"], "#333")
        ax.barh(row["Modelo"], row["ROC-AUC Macro"],
                color=color, alpha=0.85, height=0.6)
        ax.text(row["ROC-AUC Macro"] + 0.004, i,
                f"{row['ROC-AUC Macro']:.4f}", va="center", fontsize=9)

    ax.axvline(BASELINE_ROC_AUC, color="gray", ls="--", alpha=0.7,
               label=f"Baseline VGG-ish ({BASELINE_ROC_AUC})")
    ax.set_xlabel("ROC-AUC Macro")
    ax.set_title("RQ1: Zero-Shot CLAP vs Fine-Tuning Supervisado", fontweight="bold")
    legend_elems = [mpatches.Patch(facecolor=v, label=k) for k, v in palette.items()]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
    plt.tight_layout()

    out = RESULTS_DIR / "fase7_rq1_comparativa.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[OK] FASE 7 → {RESULTS_DIR / 'fase7_rq1_comparativa.csv'}")
    print(f"[OK] Figura  → {out}")
    plt.close()

    return r_clap


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — TABLA GLOBAL FINAL
# ══════════════════════════════════════════════════════════════════════════════

def tabla_global_final(etiquetas: list[str]) -> pd.DataFrame:
    """
    Construye la tabla comparativa definitiva del TFG a partir de los .npy
    guardados por las fases anteriores.

    Esta es la tabla que va directamente en la Sección de Resultados de la
    memoria. Incluye todos los modelos vs el baseline VGG-ish de MediaEval.
    """
    print("\n" + "▓" * 62)
    print("  TABLA GLOBAL — Comparativa Definitiva del TFG")
    print("▓" * 62)

    entries = [
        # (ruta_y_true, ruta_y_score, nombre, paradigma)
        (RESULTS_DIR / "panns_test_y_true.npy",
         RESULTS_DIR / "panns_test_y_score.npy",
         "PANNs CNN14", "Fine-Tuning Supervisado"),
        (RESULTS_DIR / "ast_test_y_true.npy",
         RESULTS_DIR / "ast_test_y_score.npy",
         "AST Transformer", "Fine-Tuning Supervisado"),
        (RESULTS_DIR / "multi_test_y_true.npy",
         RESULTS_DIR / "multi_test_y_score.npy",
         "Late Fusion Multimodal", "Fine-Tuning Supervisado"),
        (CLAP_Y_TRUE_PATH, CLAP_Y_SCORE_PATH,
         "CLAP", "Zero-Shot"),
    ]

    rows = [{
        "Modelo": "VGG-ish Baseline (MediaEval 2021)",
        "Paradigma": "Supervisado (referencia)",
        "ROC-AUC Macro": BASELINE_ROC_AUC,
        "PR-AUC Macro": "–", "F1 Macro (opt)": "–",
        "LRAP": "–", "Hamming Loss": "–", "Delta_Baseline": 0.0,
    }]

    for t_path, s_path, nombre, paradigma in entries:
        if t_path.exists() and s_path.exists():
            r = compute_all_metrics(
                np.load(t_path), np.load(s_path), nombre, etiquetas)
            rows.append({**r, "Paradigma": paradigma})
        else:
            rows.append({
                "Modelo": nombre, "Paradigma": paradigma,
                "ROC-AUC Macro": "(pendiente)",
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tabla_global_final.csv", index=False)

    cols_show = ["Modelo", "Paradigma", "ROC-AUC Macro",
                 "PR-AUC Macro", "F1 Macro (opt)", "LRAP", "Delta_Baseline"]
    cols_show = [c for c in cols_show if c in df.columns]
    print("\n" + df[cols_show].to_string(index=False))
    print(f"\n[OK] Tabla global → {RESULTS_DIR / 'tabla_global_final.csv'}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación final TFG — Reconocimiento de Emociones Musicales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python evaluacion_final.py --fase all       # ejecutar todo
  python evaluacion_final.py --fase 4 5       # solo FASE 4 y 5
  python evaluacion_final.py --fase 6a        # solo ablación multimodal
  python evaluacion_final.py --fase 7 tabla   # comparativa RQ1 + tabla global
        """,
    )
    parser.add_argument(
        "--fase", nargs="+",
        choices=["4", "5", "6a", "7", "tabla", "all"],
        default=["all"],
        help="Fases a ejecutar (default: all)",
    )
    args   = parser.parse_args()
    run_all = "all" in args.fase

    set_seed(SEED)
    sys.path.insert(0, str(SCRIPT_DIR))

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   TFG — Script de Evaluación Final (Emociones Musicales)     ║")
    print(f"║   Dispositivo: {str(DEVICE):<45} ║")
    print(f"║   Resultados → {str(RESULTS_DIR):<43} ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # CORRECCIÓN DEFINITIVA: las 3 etiquetas restantes del benchmark NO aparecen
    # en ninguno de los 3 splits porque los tracks que las contenían fueron
    # excluidos durante la creación del split-0. La única fuente con las 59
    # etiquetas es autotagging_moodtheme.tsv (el TSV completo del dataset).
    print("\nCargando vocabulario completo del benchmark (autotagging_moodtheme.tsv)...")
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
    etiquetas = sorted(_todas)
    print(f"Vocabulario: {len(etiquetas)} etiquetas")  # → 59

    df_test = load_split_tsv(TEST_TSV)
    print(f"Test set: {len(df_test)} canciones")

    if run_all or "4" in args.fase:
        fase4_metricas_finales(etiquetas, df_test)

    if run_all or "5" in args.fase:
        fase5_robustez_temporal(etiquetas, df_test)
        plot_temporal_drop()

    if run_all or "6a" in args.fase:
        fase6a_ablacion_multimodal(etiquetas)

    if run_all or "7" in args.fase:
        fase7_zero_shot(etiquetas, df_test)

    if run_all or "tabla" in args.fase:
        tabla_global_final(etiquetas)

    print(f"\n{'═'*62}")
    print(f"  DONE — Todos los artefactos en: {RESULTS_DIR}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
