"""
audio_utils.py — Módulo de preprocesamiento de audio
=====================================================
TFG: Reconocimiento de Emociones Musicales (Fine-Tuning vs Zero-Shot)

Clase principal: AudioPreprocessor
  - Carga de audio (mono, target_sr)
  - Ajuste de duración a 30 s exactos (pad / crop)
  - Normalización de pico
  - Extracción de Log-Mel Spectrogram (compatible con PANNs)
"""

from __future__ import annotations

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioPreprocessor:
    """Pipeline de preprocesamiento de audio para modelos de audio tagging.

    Parameters
    ----------
    target_sr : int
        Frecuencia de muestreo objetivo. Por defecto 32 000 Hz (PANNs).
    duration : float
        Duración objetivo en segundos. Por defecto 30.0 s.
    n_mels : int
        Número de bandas Mel para el espectrograma. Por defecto 128.
    n_fft : int
        Tamaño de la FFT. Por defecto 1024.
    hop_length : int
        Salto entre ventanas. Por defecto 320 (~10 ms a 32 kHz).
    """

    def __init__(
        self,
        target_sr: int = 32_000,
        duration: float = 30.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
    ) -> None:
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_samples = int(target_sr * duration)

    # ── Carga ───────────────────────────────────────────────────────────
    def load_audio(self, path: str) -> np.ndarray:
        """Carga un archivo de audio y lo convierte a mono con el sr objetivo.

        Parameters
        ----------
        path : str
            Ruta al archivo de audio (mp3, wav, flac, …).

        Returns
        -------
        np.ndarray
            Señal mono con shape ``(n_samples,)``.

        Notes
        -----
        NO se aplica librosa.effects.trim intencionalmente.
        El trim desplaza el origen temporal de forma variable por canción,
        lo que destruye la garantía de que los segundos 0-30 siempre corresponden
        al mismo segmento. Esto es crítico para RQ3 (experimento de robustez
        temporal): al evaluar el segmento 30-60s debemos asegurar que el
        desplazamiento es exactamente +30s, sin ambigüedad.
        La normalización de longitud la gestiona _fix_length() de forma
        determinista (zero-pad o crop desde el inicio del archivo original).
        """
        y, _ = librosa.load(path, sr=self.target_sr, mono=True)
        return y

    # ── Ajuste de duración ──────────────────────────────────────────────
    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        """Ajusta la señal a exactamente ``target_samples`` muestras.

        - Si es más corta → padding con ceros (zero-pad al final).
        - Si es más larga → recorte (crop) desde el inicio.

        Parameters
        ----------
        y : np.ndarray
            Señal de audio mono.

        Returns
        -------
        np.ndarray
            Señal con longitud exacta ``self.target_samples``.
        """
        n = self.target_samples
        if len(y) < n:
            # Zero-pad al final
            y = np.pad(y, (0, n - len(y)), mode="constant")
        elif len(y) > n:
            # Crop desde el inicio
            y = y[:n]
        return y

    # ── Normalización ───────────────────────────────────────────────────
    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalización de pico: escala el audio para que el valor
        absoluto máximo sea 1.0.

        Parameters
        ----------
        y : np.ndarray
            Señal de audio.

        Returns
        -------
        np.ndarray
            Señal normalizada.
        """
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    # ── Log-Mel Spectrogram ─────────────────────────────────────────────
    def get_melspec(self, y: np.ndarray) -> np.ndarray:
        """Calcula el Log-Mel Spectrogram de una señal de audio.

        Parameters
        ----------
        y : np.ndarray
            Señal de audio mono (ya preprocesada: mono, fija, normalizada).

        Returns
        -------
        np.ndarray
            Log-Mel Spectrogram con shape ``(n_mels, T)``,
            donde ``T = target_samples // hop_length + 1``.
        """
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        # Conversión a escala logarítmica (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S

    # ── Pipeline completo ───────────────────────────────────────────────
    def process(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Pipeline completo: carga → fix_length → normalize → melspec.

        Parameters
        ----------
        path : str
            Ruta al archivo de audio.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(waveform, log_mel_spectrogram)``
        """
        y = self.load_audio(path)
        y = self._fix_length(y)
        y = self._normalize(y)
        log_S = self.get_melspec(y)
        return y, log_S

    # ── Visualización ───────────────────────────────────────────────────
    def plot_spectrogram(
        self,
        log_S: np.ndarray,
        title: str = "Log-Mel Spectrogram",
        figsize: tuple[int, int] = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Visualiza un Log-Mel Spectrogram.

        Parameters
        ----------
        log_S : np.ndarray
            Log-Mel Spectrogram de shape ``(n_mels, T)``.
        title : str
            Título del gráfico.
        figsize : tuple
            Tamaño de la figura.
        save_path : str or None
            Si se proporciona, guarda la figura en esta ruta.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            log_S,
            sr=self.target_sr,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
