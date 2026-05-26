"""
models_ast.py — Arquitectura Visual AST para Fine-Tuning
========================================================
Carga el Transformer pre-entrenado del MIT y adapta 
su cabeza clasificadora para las 59 emociones de Jamendo.
"""

import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

def build_ast_model(num_classes=59, device='cuda'):
    print("Construyendo arquitectura AST (Audio Spectrogram Transformer)...")
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    # Cargamos el feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    
    # Cargamos el modelo pre-entrenado cambiando la cabeza a 59 clases
    # ignore_mismatched_sizes=True es la magia que elimina las 527 clases de AudioSet
    model = ASTForAudioClassification.from_pretrained(
        model_name, 
        num_labels=num_classes,
        ignore_mismatched_sizes=True 
    ).to(device)
    
    print("AST listo para Fine-Tuning en MTG-Jamendo.")
    return model, feature_extractor