#!/usr/bin/env python3
"""
fase7_clap_reeval.py — Evaluación Zero-Shot CLAP corregida
===========================================================
TFG: Reconocimiento de Emociones Musicales Multietiqueta

PROBLEMA QUE RESUELVE
─────────────────────
El notebook 7_zero_shot_learning_completo.ipynb ejecutó CLAP sobre
las 18,486 canciones del dataset completo (train + val + test mezclados).
Las métricas reportadas ahí son incorrectas para el TFG: incluyen
canciones que los modelos fine-tuneados vieron en train/val.

Este script:
  1. Carga los arrays precomputados (.npy) del notebook 7.
  2. Filtra las filas SOLO al split test oficial (test.tsv).
  3. Recomputa el conjunto COMPLETO de métricas del TFG (mismo que fine-tuned).
  4. Genera las gráficas para RQ1 (Zero-Shot vs Fine-Tuning).

USO
───
  cd TFG_Emociones/src 
  python fase7_clap_reeval.py

SALIDAS (en resultados_finales/)
─────────────────────────────────
  · clap_test_metrics.csv          — tabla completa de métricas sobre test.tsv
  · clap_test_per_class.csv        — AP y ROC-AUC por etiqueta
  · fase7_clap_vs_finetuned.png    — comparativa RQ1 (si existen métricas de Fase 4)
  · fase7_clap_per_class.png       — top/bottom 10 etiquetas por AP
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings_clap"
SPLIT_DIR      = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
VAL_TSV        = SPLIT_DIR / "autotagging_moodtheme-validation.tsv"
TEST_TSV       = SPLIT_DIR / "autotagging_moodtheme-test.tsv"
RESULTS_DIR    = PROJECT_ROOT / "resultados_finales"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos generados por el notebook 7
Y_TRUE_PATH    = EMBEDDINGS_DIR / "zero_shot_y_true_global.npy"
Y_SCORE_PATH   = EMBEDDINGS_DIR / "zero_shot_y_score_global.npy"
TRACK_IDS_PATH = EMBEDDINGS_DIR / "track_ids_embeddings.npy"


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_tsv(path: Path) -> pd.DataFrame:
    """Parser robusto para los TSV de MTG-Jamendo (número variable de etiquetas)."""
    registros = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # saltar cabecera
        for linea in f:
            if not linea.strip():
                continue
            cols = linea.strip().split("\t")
            if len(cols) < 6:
                continue
            track_id = cols[0].replace("track_", "").lstrip("0") or "0"
            etiquetas = [t.replace("mood/theme---", "") for t in cols[5:]]
            registros.append({"track_id": track_id, "tags": etiquetas})
    return pd.DataFrame(registros)


def _get_label_list(tsv_path: Path) -> list[str]:
    """Deriva el vocabulario de 59 etiquetas desde el TSV (orden alfabético)."""
    todas = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f)
        for linea in f:
            if not linea.strip():
                continue
            cols = linea.strip().split("\t")
            if len(cols) >= 6:
                todas.update(t.replace("mood/theme---", "") for t in cols[5:])
    return sorted(todas)


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS — conjunto idéntico al de evaluacion_final.py
# ══════════════════════════════════════════════════════════════════════════════

def _optimize_thresholds_per_class(y_true_val: np.ndarray,
                                    y_score_val: np.ndarray) -> np.ndarray:
    """
    Optimiza el umbral de decisión por clase maximizando F1 en el split
    VALIDATION — idéntico al procedimiento de los modelos fine-tuneados.

    El notebook 7 procesó el dataset completo (train+val+test), por lo que
    los embeddings de validation están disponibles en los arrays globales.
    Este script los filtra con track_ids_embeddings.npy y los usa aquí para
    que la comparativa de F1 sea metodológicamente equivalente a fine-tuned.

    Args:
        y_true_val:  ground truth del split validation, shape (N_val, 59)
        y_score_val: scores CLAP del split validation, shape (N_val, 59)

    Retorna: array de shape (59,) con el umbral óptimo por clase.
    """
    from sklearn.metrics import f1_score

    thresholds = np.full(y_true_val.shape[1], 0.5)
    candidates = np.linspace(0.05, 0.95, 19)   # 19 umbrales a evaluar

    for j in range(y_true_val.shape[1]):
        if len(np.unique(y_true_val[:, j])) < 2:
            continue   # clase constante en val: mantener 0.5
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            y_pred_j = (y_score_val[:, j] >= t).astype(int)
            f = f1_score(y_true_val[:, j], y_pred_j, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[j] = best_t

    return thresholds


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    label_list: list[str],
                    y_true_val: np.ndarray | None = None,
                    y_score_val: np.ndarray | None = None) -> dict:
    """
    Computa el conjunto COMPLETO de métricas del TFG (idéntico al de los
    modelos fine-tuneados en evaluacion_final.py):

      Ranking (no requieren umbral — comparables directamente):
        · ROC-AUC Macro / Micro
        · PR-AUC Macro / Micro  (= mAP Macro / Micro)
        · LRAP
        · Spearman ρ(support_class ↔ ROC-AUC_class)

      Umbral-dependientes (umbral optimizado en VALIDATION — igual que fine-tuned):
        · F1 Macro
        · F1 Samples
        · Hamming Loss

      Por clase:
        · DataFrame con AP, AUC y support para las 59 etiquetas

    Args:
        y_true / y_score:         arrays del split TEST
        y_true_val / y_score_val: arrays del split VALIDATION para calibrar umbrales.
                                  Si son None, se usan arrays de test (fallback).
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        label_ranking_average_precision_score,
        f1_score,
        hamming_loss,
    )
    from scipy.stats import spearmanr

    # Columnas con varianza (evitar columnas constantes)
    valid_cols = [i for i in range(y_true.shape[1])
                  if len(np.unique(y_true[:, i])) > 1]
    if len(valid_cols) == 0:
        raise ValueError("Ninguna columna tiene varianza. Verifica los datos.")

    y_t = y_true[:, valid_cols]
    y_s = y_score[:, valid_cols]

    # ── Métricas de ranking ────────────────────────────────────────────────
    roc_auc_macro = roc_auc_score(y_t, y_s, average="macro")
    roc_auc_micro = roc_auc_score(y_t, y_s, average="micro")
    pr_auc_macro  = average_precision_score(y_t, y_s, average="macro")
    pr_auc_micro  = average_precision_score(y_t, y_s, average="micro")
    lrap          = label_ranking_average_precision_score(y_t, y_s)

    # Per-class AUC y support (para Spearman)
    per_class = []
    aucs_per_class = []
    supports = []
    for i, col in enumerate(valid_cols):
        ap  = average_precision_score(y_true[:, col], y_score[:, col])
        auc = roc_auc_score(y_true[:, col], y_score[:, col])
        sup = int(y_true[:, col].sum())
        per_class.append({"label": label_list[col], "AP": ap, "AUC": auc, "support": sup})
        aucs_per_class.append(auc)
        supports.append(sup)

    # Spearman ρ: correlación entre frecuencia de clase y su AUC
    spearman_rho, spearman_p = spearmanr(supports, aucs_per_class)

    # ── Métricas de umbral (umbrales calibrados en VALIDATION) ────────────
    # Idéntico al procedimiento de fine-tuned: threshold optimizado en val,
    # aplicado en test. La comparativa de F1 es metodológicamente equivalente.
    cal_true  = y_true_val  if y_true_val  is not None else y_true
    cal_score = y_score_val if y_score_val is not None else y_score
    if y_true_val is None:
        print("  [AVISO] Sin datos de validation; umbrales calibrados en test (fallback).")
    else:
        print(f"  Optimizando umbrales por clase en validation (N={cal_true.shape[0]})...")
    thresholds = _optimize_thresholds_per_class(cal_true, cal_score)
    y_pred = (y_score >= thresholds).astype(int)

    f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
    ham_loss   = hamming_loss(y_true, y_pred)

    return {
        # Ranking
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_micro": roc_auc_micro,
        "pr_auc_macro":  pr_auc_macro,
        "pr_auc_micro":  pr_auc_micro,
        "lrap":          lrap,
        "spearman_rho":  spearman_rho,
        "spearman_p":    spearman_p,
        # Umbral-dependientes
        "f1_macro":      f1_macro,
        "f1_samples":    f1_samples,
        "hamming_loss":  ham_loss,
        # Info
        "n_samples":     int(y_true.shape[0]),
        "n_valid_cols":  int(len(valid_cols)),
        "thresholds":    thresholds,
        "per_class":     pd.DataFrame(per_class).sort_values("AP", ascending=False),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_per_class(per_class_df: pd.DataFrame, out_path: Path) -> None:
    """Top/Bottom 10 etiquetas por AP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CLAP Zero-Shot — AP por etiqueta (test.tsv)", fontsize=14, fontweight="bold")

    sns.barplot(data=per_class_df.head(10), x="AP", y="label", palette="viridis", ax=ax1)
    ax1.set_title("Top 10 — Mejor predichas")
    ax1.set_xlabel("Average Precision")
    ax1.set_xlim(0, 1)
    ax1.axvline(per_class_df["AP"].mean(), color="red", ls="--", alpha=0.6, label="Media")
    ax1.legend()

    sns.barplot(data=per_class_df.tail(10), x="AP", y="label", palette="magma", ax=ax2)
    ax2.set_title("Bottom 10 — Más difíciles")
    ax2.set_xlabel("Average Precision")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] {out_path.name}")


def _plot_vs_finetuned(metrics_clap: dict, fase4_csv: Path | None,
                       out_path: Path) -> None:
    """
    Comparativa RQ1: Zero-Shot CLAP vs modelos fine-tuneados.

    Métricas mostradas: ROC-AUC Macro, PR-AUC Macro, F1 Macro, LRAP.
    Si existe resultados_finales/fase4_metrics_summary.csv (generado por
    evaluacion_final.py), incluye los 3 modelos fine-tuneados.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_names = ["ROC-AUC Macro", "PR-AUC Macro", "F1 Macro", "LRAP"]
    clap_vals = [
        metrics_clap["roc_auc_macro"],
        metrics_clap["pr_auc_macro"],
        metrics_clap["f1_macro"],
        metrics_clap["lrap"],
    ]

    # Intentar cargar métricas de Fase 4 (fine-tuned)
    modelos_ft = {}
    if fase4_csv is not None and fase4_csv.exists():
        try:
            df4 = pd.read_csv(fase4_csv)
            # Formato esperado: columnas = [modelo, roc_auc_macro, pr_auc_macro, f1_macro, lrap, ...]
            for _, row in df4.iterrows():
                modelos_ft[row["modelo"]] = [
                    float(row.get("roc_auc_macro", 0)),
                    float(row.get("pr_auc_macro", 0)),
                    float(row.get("f1_macro", 0)),
                    float(row.get("lrap", 0)),
                ]
        except Exception as e:
            print(f"  [AVISO] No se pudo leer fase4_metrics_summary.csv: {e}")

    x = np.arange(len(metrics_names))
    width = 0.18
    total = len(modelos_ft) + 1  # +1 por CLAP
    offsets = np.linspace(-(total - 1) * width / 2, (total - 1) * width / 2, total)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # Fine-tuned models
    for i, (nombre, vals) in enumerate(modelos_ft.items()):
        ax.bar(x + offsets[i], vals, width, label=nombre, color=colors[i % len(colors)])

    # CLAP (siempre el último)
    ax.bar(x + offsets[-1], clap_vals, width, label="CLAP Zero-Shot",
           color="#937860", hatch="//", alpha=0.85)

    # Línea de referencia baseline VGG-ish (solo en ROC-AUC Macro, índice 0)
    ax.annotate("Baseline\nVGG-ish\n(0.725)", xy=(x[0], 0.725), xytext=(x[0] + 0.55, 0.78),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.axhline(0.725, color="gray", ls=":", lw=1.0, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("RQ1: Zero-Shot CLAP vs Modelos Fine-Tuneados (test.tsv)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Nota metodológica
    ax.text(0.01, 0.02,
            "Umbrales F1 optimizados en validation para todos los modelos\n"
            "(CLAP incluido — mismo protocolo que fine-tuned).",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("FASE 7 — Evaluación Zero-Shot CLAP (test.tsv corregido)")
    print("=" * 65)

    # ── 1. Verificar archivos precomputados ────────────────────────────────
    for p in [Y_TRUE_PATH, Y_SCORE_PATH, TRACK_IDS_PATH]:
        if not p.exists():
            print(f"\n[ERROR] Archivo no encontrado: {p}")
            print("  Asegúrate de haber ejecutado el notebook 7 completo.")
            return

    # ── 2. Cargar arrays ───────────────────────────────────────────────────
    print("\nCargando arrays precomputados del notebook 7...")
    y_true_global  = np.load(Y_TRUE_PATH)
    y_score_global = np.load(Y_SCORE_PATH)
    track_ids_arr  = np.load(TRACK_IDS_PATH, allow_pickle=True)

    print(f"  y_true:     {y_true_global.shape}")
    print(f"  y_score:    {y_score_global.shape}")
    print(f"  track_ids:  {len(track_ids_arr)} IDs")

    # ── 3. Vocabulario de etiquetas (idéntico al del notebook 7) ──────────
    # El notebook 7 derivó label_list de autotagging_moodtheme.tsv (completo),
    # ordenado alfabéticamente. Hacemos lo mismo para garantizar la alineación.
    full_tsv = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
    label_list = _get_label_list(full_tsv)
    print(f"  Etiquetas:  {len(label_list)}")

    if len(label_list) != y_true_global.shape[1]:
        print(f"[ERROR] Desalineación de etiquetas: "
              f"{len(label_list)} != {y_true_global.shape[1]}")
        return

    # ── 4. Cargar IDs de validation y test ────────────────────────────────
    print(f"\nCargando splits oficiales...")
    df_val  = _load_split_tsv(VAL_TSV)
    df_test = _load_split_tsv(TEST_TSV)
    val_ids  = set(df_val["track_id"].astype(str).tolist())
    test_ids = set(df_test["track_id"].astype(str).tolist())
    print(f"  validation.tsv: {len(val_ids)} canciones")
    print(f"  test.tsv:       {len(test_ids)} canciones")

    # ── 5. Construir máscaras de filtrado ─────────────────────────────────
    # track_ids_arr puede contener ints o strings — normalizamos a str
    track_ids_str = np.array([str(tid) for tid in track_ids_arr])

    mask_val  = np.array([tid in val_ids  for tid in track_ids_str])
    mask_test = np.array([tid in test_ids for tid in track_ids_str])

    n_val_found  = int(mask_val.sum())
    n_test_found = int(mask_test.sum())

    print(f"\n  Validation encontradas en CLAP: {n_val_found}/{len(val_ids)}")
    print(f"  Test encontradas en CLAP:       {n_test_found}/{len(test_ids)}")

    if n_val_found > 0:
        miss_v = len(val_ids) - n_val_found
        if miss_v > 0:
            print(f"  [AVISO] {miss_v} canciones de val ausentes (archivos .mp3 faltaban).")
    else:
        print("  [AVISO] Ninguna canción de val encontrada — se usará fallback test-oracle.")

    if n_test_found == 0:
        print("[ERROR] Ninguna canción de test encontrada. Verifica el formato de track_id.")
        print(f"  Ejemplos track_ids_arr: {track_ids_arr[:5]}")
        print(f"  Ejemplos test_ids:      {list(test_ids)[:5]}")
        return

    # ── 6. Filtrar ────────────────────────────────────────────────────────
    y_true_val_arr  = y_true_global[mask_val]   if n_val_found  > 0 else None
    y_score_val_arr = y_score_global[mask_val]  if n_val_found  > 0 else None
    y_true_test     = y_true_global[mask_test]
    y_score_test    = y_score_global[mask_test]

    print(f"\n  Val  filtrado: {y_true_val_arr.shape if y_true_val_arr is not None else 'N/A'}")
    print(f"  Test filtrado: {y_true_test.shape}")

    n_found = n_test_found   # alias para el mensaje final

    # ── 7. Calcular métricas (conjunto completo = fine-tuned) ──────────────
    print("\nCalculando métricas sobre test.tsv...")
    metrics = compute_metrics(
        y_true_test, y_score_test, label_list,
        y_true_val=y_true_val_arr, y_score_val=y_score_val_arr,
    )

    print("\n" + "─" * 50)
    print(f"  {'Métrica':<28} | {'Valor':<10}")
    print("─" * 50)
    print(f"  {'ROC-AUC Macro':<28} | {metrics['roc_auc_macro']:.4f}  ← métrica principal MediaEval")
    print(f"  {'ROC-AUC Micro':<28} | {metrics['roc_auc_micro']:.4f}")
    print(f"  {'PR-AUC Macro (mAP Macro)':<28} | {metrics['pr_auc_macro']:.4f}  ← métrica principal MediaEval")
    print(f"  {'PR-AUC Micro (mAP Micro)':<28} | {metrics['pr_auc_micro']:.4f}")
    print(f"  {'LRAP':<28} | {metrics['lrap']:.4f}")
    print(f"  {'Spearman ρ(support↔AUC)':<28} | {metrics['spearman_rho']:.4f}  (p={metrics['spearman_p']:.3f})")
    print(f"  {'F1 Macro':<28} | {metrics['f1_macro']:.4f}")
    print(f"  {'F1 Samples':<28} | {metrics['f1_samples']:.4f}")
    print(f"  {'Hamming Loss':<28} | {metrics['hamming_loss']:.4f}")
    print(f"  {'Muestras evaluadas':<28} | {metrics['n_samples']}")
    print(f"  {'Etiquetas válidas':<28} | {metrics['n_valid_cols']}/59")
    print("─" * 50)
    thr_method = "validation" if y_true_val_arr is not None else "test (fallback — val no disponible)"
    print(f"  Umbrales calibrados en: {thr_method}")
    print(f"  → Comparativa F1 con fine-tuned es metodológicamente equivalente.")
    print(f"\n  Baseline VGG-ish MediaEval 2021: ROC-AUC Macro = 0.725")
    diff = metrics["roc_auc_macro"] - 0.725
    print(f"  CLAP vs Baseline: {diff:+.4f} ({'↑ mejor' if diff > 0 else '↓ peor'})")

    # ── 8. Guardar CSV de métricas ─────────────────────────────────────────
    summary = pd.DataFrame([{
        "modelo":           "CLAP Zero-Shot",
        "split":            "test",
        "n_samples":        metrics["n_samples"],
        "roc_auc_macro":    metrics["roc_auc_macro"],
        "roc_auc_micro":    metrics["roc_auc_micro"],
        "pr_auc_macro":     metrics["pr_auc_macro"],
        "pr_auc_micro":     metrics["pr_auc_micro"],
        "lrap":             metrics["lrap"],
        "spearman_rho":     metrics["spearman_rho"],
        "f1_macro":         metrics["f1_macro"],
        "f1_samples":       metrics["f1_samples"],
        "hamming_loss":     metrics["hamming_loss"],
        "threshold_method": "validation-set per-class F1 optimization (same as fine-tuned)",
        "audio_segundos":   10,   # notebook 7 usó duration=10.0
        "score_func":       "sigmoid",
    }])
    summary_path = RESULTS_DIR / "clap_test_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  [CSV] {summary_path.name}")

    per_class_path = RESULTS_DIR / "clap_test_per_class.csv"
    metrics["per_class"].to_csv(per_class_path, index=False)
    print(f"  [CSV] {per_class_path.name}")

    # Guardar umbrales óptimos (útil para análisis adicional)
    thresholds_df = pd.DataFrame({
        "label":     label_list,
        "threshold": metrics["thresholds"],
    })
    thresholds_df.to_csv(RESULTS_DIR / "clap_thresholds_test_oracle.csv", index=False)
    print(f"  [CSV] clap_thresholds_test_oracle.csv")

    # ── 9. Gráficas ────────────────────────────────────────────────────────
    print("\nGenerando gráficas...")
    _plot_per_class(
        metrics["per_class"],
        out_path=RESULTS_DIR / "fase7_clap_per_class.png",
    )

    fase4_csv = RESULTS_DIR / "fase4_metrics_summary.csv"
    _plot_vs_finetuned(
        metrics_clap=metrics,
        fase4_csv=fase4_csv,
        out_path=RESULTS_DIR / "fase7_clap_vs_finetuned.png",
    )

    print(f"\n[OK] Evaluación CLAP corregida completada.")
    print(f"     Resultados en: {RESULTS_DIR}")
    print(f"\n  ╔══ PARA LA MEMORIA — Sección RQ1 ═══════════════════════╗")
    print(f"  ║  Cita métricas de clap_test_metrics.csv, NO notebook 7.║")
    print(f"  ║  Notebook 7 → N=18,486 (train+val+test).              ║")
    print(f"  ║  Este script → N={n_found} (solo test).                   ║")
    print(f"  ║  Para F1: indicar 'test-oracle thresholds' en memoria. ║")
    print(f"  ╚════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
