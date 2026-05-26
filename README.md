# Music Emotion Recognition (MER): Fine-Tuning vs. Zero-Shot Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Repositorio oficial del Trabajo Fin de Grado en Ciencia e Ingeniería de Datos (Universidad de Murcia), enfocado en el Reconocimiento Automático de Emociones Musicales sobre el corpus [MTG-Jamendo Mood/Theme](https://mtg.github.io/mtg-jamendo-dataset/).

## Resumen del Proyecto

Este proyecto aborda la tarea de MER como un problema de clasificación multietiqueta (59 etiquetas emocionales/temáticas), comparando paradigmas de aprendizaje supervisado (Fine-Tuning de CNNs y Transformers) frente a enfoques Zero-Shot (CLAP). 

Se ha diseñado un modelo Multimodal de Fusión Tardía (Late Fusion) que combina representaciones acústicas (PANNs), espectrales (AST) y semánticas (Whisper), superando el baseline oficial del reto MediaEval 2021.

### Aportaciones Técnicas Destacadas

* **Ingeniería para GPU Doméstica:** Entrenamiento de modelos masivos (AST, PANNs) en hardware de consumo utilizando precisión mixta bfloat16, acumulación de gradientes y corrección de alineamiento en el scheduler OneCycleLR (uso de math.ceil()).
* **Prevención de Data Leakage:** Uso estricto del Split-0 para evitar el Efecto Artista.
* **Corrección de Vocabulario (Bugfix):** Implementación de un pipeline de normalización que reconstruye el vocabulario completo de 59 etiquetas a partir de los metadatos maestros, solucionando la omisión de 3 etiquetas en los TSVs oficiales.
* **Evaluación de Robustez Temporal:** Análisis del sesgo posicional evaluando ventanas temporales no vistas (30-60s) sin reentrenamiento.

---

## Resultados Principales (Test Split-0)

| Modelo | Arquitectura | ROC-AUC Macro | PR-AUC Macro | F1 (opt.) | Δ vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| MediaEval 2021 | VGG-ish (Baseline) | 0.7250 | - | - | 0.00 |
| **Late Fusion** | **AST + PANNs + Whisper** | **0.7518** | **0.1382** | **0.1799** | **+0.0268** |
| PANNs CNN14 | CNN | 0.7471 | 0.1293 | 0.1800 | +0.0221 |
| AST | Transformer | 0.7197 | 0.1162 | 0.1619 | -0.0053 |
| CLAP | Zero-Shot (Audio-Text) | 0.6412 | 0.0593 | 0.0541 | -0.0838 |

---

## Estructura del Repositorio

El código fuente está modularizado para separar el procesamiento de audio, la definición de arquitecturas y los bucles de entrenamiento.

```text
├── data/                     # Audio crudo y embeddings preextraídos (ignorado en Git)
├── figures/                  # Figuras del análisis exploratorio del corpus
├── memoria_latex/            # Código fuente modular de la memoria del TFG en LaTeX
├── models/                   # Checkpoints (.pth) de los modelos entrenados
├── mtg-jamendo-dataset/      # Submódulo Git con los TSV del Split-0 oficial
├── notebooks/                # Cuadernos de análisis interactivo y extracción offline
│   ├── analisis_exploratorio.ipynb      # EDA y distribución de etiquetas
│   ├── extraccion_ast.ipynb             # Extracción de características temporales AST
│   ├── extraccion_panns.ipynb           # Extracción de características acústicas
│   ├── extraccion_whisper.ipynb         # Extracción de características semánticas
│   └── ...                              # (Auditorías, pruebas GPU y análisis por clase)
├── resultados_finales/       # Archivos CSV y PNGs de evaluación generados (Fases 4-6)
└── src/                      # Scripts principales de ejecución y módulos base
    ├── models.py             # Definición de MultimodalLateFusion y adaptadores
    ├── train_panns.py        # Entrenamiento de la arquitectura CNN14
    ├── train_ast.py          # Entrenamiento del Transformer (AST)
    ├── train_multimodal.py   # Entrenamiento del MLP de Fusión Tardía
    ├── evaluacion_final.py   # Cálculo de métricas, matrices y curvas ROC
    └── ...                   # (Módulo de dataset, utilidades de audio y zero-shot)
