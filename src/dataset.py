"""
dataset.py — Dataset de PyTorch para MTG-Jamendo (Mood/Theme)
==============================================================
TFG: Reconocimiento de Emociones Musicales (Fine-Tuning vs Zero-Shot)

Clase principal: MTGJamendoDataset
  - Carga de audio y extracción de Log-Mel Spectrogram (Para Baseline)
  - Carga de audio crudo (Para PANNs)
  - Codificación multi-label (one-hot, 59 clases)

Función auxiliar: load_split_tsv
  - Carga un TSV oficial del split-0 de MTG-Jamendo (train/validation/test)
  - CRÍTICO: Elimina el riesgo de Data Leakage / Efecto Artista
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from audio_utils import AudioPreprocessor


# ==============================================================
# FUNCIÓN CLAVE: Carga de Splits Oficiales (Anti Data-Leakage)
# ==============================================================

def load_split_tsv(tsv_path: str | Path) -> pd.DataFrame:
    """Carga un TSV oficial del split-0 de MTG-Jamendo y devuelve un DataFrame.

    Utilizar SIEMPRE esta función en lugar de df.sample() para garantizar
    que no hay "Efecto Artista" (Artist Leakage) entre train/val/test.
    El split-0 de MTG-Jamendo asegura que canciones del mismo artista
    NO caen en particiones distintas.

    Parameters
    ----------
    tsv_path : str | Path
        Ruta al archivo TSV de split-0 (train, validation o test).
        Ejemplo: 'mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv'

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ['path', 'tags'] donde 'tags' es una lista
        de strings con las etiquetas mood/theme ya limpias (sin prefijo).
    """
    tsv_path = Path(tsv_path)
    registros = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        next(f)  # Saltar cabecera: TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS
        for linea in f:
            if not linea.strip():
                continue
            parts = linea.strip().split('\t')
            if len(parts) >= 6:
                # Filtrar solo etiquetas mood/theme y eliminar el prefijo
                tags = [
                    t.replace('mood/theme---', '').strip()
                    for t in parts[5:]
                    if t.startswith('mood/theme---')
                ]
                if tags:
                    registros.append({'path': parts[3], 'tags': tags})
    return pd.DataFrame(registros)

class MTGJamendoDataset(Dataset):
    """Dataset de PyTorch para el subset mood/theme de MTG-Jamendo."""

    def __init__(
        self,
        df,
        audio_dir: str | Path,
        preprocessor: AudioPreprocessor,
        label_list: list[str],
        return_raw: bool = False # <-- ¡NUEVO PARÁMETRO!
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.preprocessor = preprocessor
        self.label_list = label_list
        self.n_classes = len(label_list)
        self.return_raw = return_raw # Guardamos la preferencia
        self.label2idx = {label: i for i, label in enumerate(label_list)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # --- LÓGICA DE FALLBACK (.low.mp3) ---
        path_limpio = str(row["path"]).strip()
        audio_path_normal = self.audio_dir / path_limpio
        audio_path_low = self.audio_dir / path_limpio.replace('.mp3', '.low.mp3')

        if audio_path_normal.exists():
            track_path = audio_path_normal
        elif audio_path_low.exists():
            track_path = audio_path_low
        else:
            track_path = audio_path_normal # Fallback por defecto

        # ── Etiquetas: one-hot encoding ──
        # ... (el resto sigue igual)

        # ── Etiquetas: one-hot encoding ──
        label_vector = torch.zeros(self.n_classes, dtype=torch.float32)
        for tag in row["tags"]:
            if tag in self.label2idx:
                label_vector[self.label2idx[tag]] = 1.0

        # ── Carga de Datos (El Bypass SOTA) ──
        if self.return_raw:
            carga = self.preprocessor.load_audio(track_path)
            audio_wave = carga[0] if isinstance(carga, tuple) else carga
            
            # --- ESTANDARIZACIÓN ESTRICTA A 30 SEGUNDOS ---
            # CORRECCIÓN: Usar target_sr DINÁMICO del preprocesador.
            # Antes era `32000 * 30` hardcodeado, lo que causaba que AST
            # (16kHz) recibiera 960000 muestras = 60 segundos de audio, no 30.
            # Ahora: 32000*30=960000 para PANNs, 16000*30=480000 para AST.
            target_length = self.preprocessor.target_sr * 30
            
            if len(audio_wave) > target_length:
                # Recortamos a los primeros 30 segundos
                audio_wave = audio_wave[:target_length]
            elif len(audio_wave) < target_length:
                # Rellenamos con silencios (ceros) al final si es más corta
                padding = target_length - len(audio_wave)
                audio_wave = np.pad(audio_wave, (0, padding), 'constant')
                
            # Forzamos una copia en memoria para que el DataLoader de Windows no se bloquee
            audio_wave = np.array(audio_wave, copy=True)
            # -----------------------------------------------------
            
            return torch.tensor(audio_wave, dtype=torch.float32), label_vector
        else:
            # PARA SIMPLECNN/AST: Extraer espectrograma log-mel con librosa (CPU)
            log_mel = self.preprocessor.process(track_path)
            spectrogram = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
            return spectrogram, label_vector

    def get_pos_weight(self) -> torch.Tensor:
        """Calcula pos_weight para contrarrestar el desbalanceo de clases."""
        n_total = len(self.df)
        pos_counts = np.zeros(self.n_classes, dtype=np.float64)

        for tags in self.df["tags"]:
            for tag in tags:
                if tag in self.label2idx:
                    pos_counts[self.label2idx[tag]] += 1

        eps = 1e-6
        pos_counts = np.maximum(pos_counts, eps)
        neg_counts = n_total - pos_counts
        weight = neg_counts / pos_counts
        return torch.tensor(weight, dtype=torch.float32)