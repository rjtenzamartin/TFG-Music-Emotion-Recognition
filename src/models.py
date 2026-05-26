"""
models.py — Modelos de clasificación de audio (Baseline)
=========================================================
TFG: Reconocimiento de Emociones Musicales (Fine-Tuning vs Zero-Shot)

Clase principal: SimpleCNN
  - 4 bloques convolucionales (32 → 64 → 128 → 256)
  - Cabezal: AdaptiveAvgPool2d → Linear(n_classes)
  - Salida sin Sigmoid (usar BCEWithLogitsLoss)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN baseline para multi-label audio tagging.

    Arquitectura
    ------------
    4 bloques idénticos:
        Conv2d(3×3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)

    Filtros por bloque: 32 → 64 → 128 → 256

    Cabezal:
        AdaptiveAvgPool2d(1, 1) → Flatten → Linear(256, n_classes)

    Parameters
    ----------
    n_classes : int
        Número de clases de salida. Por defecto 59 (mood/theme).
    in_channels : int
        Canales de entrada. Por defecto 1 (espectrograma monocanal).
    """

    def __init__(self, n_classes: int = 59, in_channels: int = 1) -> None:
        super().__init__()

        # ── Bloques convolucionales ──
        channels = [in_channels, 32, 64, 128, 256]

        blocks = []
        for i in range(4):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1],
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
        self.features = nn.Sequential(*blocks)

        # ── Cabezal de clasificación ──
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)
        # Sin Sigmoid: usamos BCEWithLogitsLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Entrada con shape ``(B, 1, n_mels, T)``.

        Returns
        -------
        torch.Tensor
            Logits con shape ``(B, n_classes)``.
        """
        x = self.features(x)       # (B, 256, H', W')
        x = self.pool(x)           # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.classifier(x)     # (B, n_classes)
        return x
