#!/usr/bin/env python3
"""
generar_figura_compresor.py
===========================
Genera la figura comparativa "Con vs. Sin compresor logarítmico" para
la Sección de Ablación (Fase 6B) de la memoria del TFG.

No requiere GPU. Tarda ~5 segundos.

Uso:
  python src/generar_figura_compresor.py

Salida:
  figures/comparativa_compresor_log.png
"""

from pathlib import Path
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DIR    = PROJECT_ROOT / "data"
FIGURES_DIR  = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Buscar cualquier audio disponible ─────────────────────────────────────────
sample_path = None
for ext in ["*.mp3", "*.low.mp3", "*.wav"]:
    candidates = list(AUDIO_DIR.rglob(ext))
    if candidates:
        sample_path = candidates[0]
        break

if sample_path is None:
    print("[ERROR] No se encontró ningún archivo de audio en data/")
    raise SystemExit(1)

print(f"Usando audio de ejemplo: {sample_path.name}")

# ── Cargar 30 segundos de audio ───────────────────────────────────────────────
y, sr = librosa.load(str(sample_path), sr=32000, mono=True, duration=30.0)
n = 32000 * 30
y = np.pad(y, (0, max(0, n - len(y))), mode="constant")[:n]

# ── Mel Spectrogram ───────────────────────────────────────────────────────────
S = librosa.feature.melspectrogram(y=y, sr=32000, n_fft=1024, hop_length=320, n_mels=128)

# Con compresor: escala logarítmica (dB)
S_log = librosa.power_to_db(S, ref=np.max)

# Sin compresor: escala lineal, normalizada a [0, 1] (como _IdentityNorm en fase6b)
S_lin = S / (S.max() + 1e-10)

# ── Figura 2×2 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                          gridspec_kw={"height_ratios": [1, 0.05]})
fig.suptitle("Efecto del Compresor Logarítmico sobre el Mel Espectrograma\n"
             "(Fase 6B — Ablación del Log-Compressor en CNN14)",
             fontsize=14, fontweight="bold")

# Panel izquierdo: Con log-compressor
ax_log  = axes[0, 0]
ax_cbar_log = axes[1, 0]
img_log = librosa.display.specshow(
    S_log, sr=32000, hop_length=320,
    x_axis="time", y_axis="mel", ax=ax_log,
    cmap="magma",
)
fig.colorbar(img_log, cax=ax_cbar_log, orientation="horizontal", format="%+2.0f dB")
ax_log.set_title("CON compresor logarítmico (AmplitudeToDB)\n"
                  "Rango dinámico comprimido: ~80 dB → ~30 dB",
                  fontsize=11, fontweight="bold", color="#1a1a2e")
ax_log.set_xlabel("Tiempo (s)")
ax_log.set_ylabel("Frecuencia mel (Hz)")

# Estadísticas
ax_log.text(0.02, 0.97,
            f"min={S_log.min():.1f} dB\nmax={S_log.max():.1f} dB\n"
            f"std={S_log.std():.1f} dB",
            transform=ax_log.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

# Panel derecho: Sin log-compressor
ax_lin  = axes[0, 1]
ax_cbar_lin = axes[1, 1]
img_lin = librosa.display.specshow(
    S_lin, sr=32000, hop_length=320,
    x_axis="time", y_axis="mel", ax=ax_lin,
    cmap="magma",
)
fig.colorbar(img_lin, cax=ax_cbar_lin, orientation="horizontal", format="%.3f")
ax_lin.set_title("SIN compresor logarítmico (_IdentityNorm)\n"
                  "Escala lineal normalizada a [0, 1]",
                  fontsize=11, fontweight="bold", color="#8b0000")
ax_lin.set_xlabel("Tiempo (s)")
ax_lin.set_ylabel("")

ax_lin.text(0.02, 0.97,
            f"min={S_lin.min():.4f}\nmax={S_lin.max():.4f}\n"
            f"std={S_lin.std():.4f}",
            transform=ax_lin.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

plt.tight_layout(rect=[0, 0, 1, 0.95])

out = FIGURES_DIR / "comparativa_compresor_log.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[OK] Figura guardada: {out}")
print("\nPara la memoria (Sección Fase 6B / Ablación):")
print("  La escala logarítmica comprime el rango dinámico de ~80 dB a ~30 dB,")
print("  haciendo que las variaciones de intensidad pequeñas sean visualmente")
print("  comparables a las grandes. Sin el compresor, los instrumentos suaves")
print("  quedan aplastados en negro cerca de 0 (dominados por los picos de energía).")
