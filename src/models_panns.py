"""
models_panns.py — Arquitectura CNN14 de PANNs (State-of-the-Art)
===============================================================
Implementación exacta del modelo ganador de AudioSet adaptada para 
el dataset MTG-Jamendo (59 emociones).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

# --- 1. CAPAS DE EXTRACCIÓN DE ESPECTROGRAMAS EN GPU ---
# (PANNs calcula el Mel-Spectrogram usando convoluciones 1D)
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class SpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, n_mels=64):
        super().__init__()
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            hop_length=hop_size,
            n_mels=n_mels,
            f_min=50,
            f_max=14000
        )
        self.db_transform = AmplitudeToDB(top_db=80)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.mel_transform(x) # (batch_size, n_mels, time_steps)
        x = self.db_transform(x)
        # PANNs espera (batch_size, channels, time_steps, n_mels)
        x = x.transpose(1, 2).unsqueeze(1) 
        return x

# --- 2. BLOQUE CONVOLUCIONAL BÁSICO DE PANNs ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2)
        return x

# --- 3. LA ARQUITECTURA CNN14 COMPLETA ---
class CNN14_Transfer(nn.Module):
    def __init__(self, num_classes=59, sample_rate=32000):
        super().__init__()
        
        # Extractor de características en GPU
        self.spectrogram_extractor = SpectrogramExtractor(sample_rate=sample_rate)
        
        # Red Profunda (14 capas)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        
        # CLASIFICADOR ORIGINAL DE AUDIOSET (Lo creamos para poder cargar los pesos)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)
        
        # NUESTRA NUEVA CABEZA (Para MTG-Jamendo)
        self.fc_jamendo = nn.Linear(2048, num_classes, bias=True)
        
        # Inicializar pesos
        self.init_weights()

    def init_weights(self):
        # Inicialización de la cabeza nueva (las convoluciones se sobrescribirán con los pesos pre-entrenados)
        nn.init.kaiming_normal_(self.fc_jamendo.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_jamendo.bias, 0)

    def forward(self, x):
        # 1. GPU calcula espectrograma: (batch_size, 1, time, mel_bins)
        x = self.spectrogram_extractor(x)
        x = x.transpose(1, 3) 
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # 2. Convoluciones
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        
        # 3. Pooling Global (PANNs usa la media en el tiempo)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2) # Mezcla de Max y Avg Pooling
        
        # 4. Capas densas (Transfer Learning)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # ¡Magia! Usamos nuestra cabeza de Jamendo en lugar de la de AudioSet
        output = self.fc_jamendo(x)
        
        return output

# --- 4. FUNCIÓN PARA DESCARGAR Y CARGAR PESOS SOTA ---
def build_panns_model(num_classes=59, device='cuda'):
    print("Construyendo arquitectura CNN14 de PANNs...")
    model = CNN14_Transfer(num_classes=num_classes).to(device)
    
    # URL oficial de los pesos pre-entrenados en AudioSet
    checkpoint_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    checkpoint_path = "Cnn14_mAP=0.431.pth"
    
    # Descargar pesos si no existen
    if not os.path.exists(checkpoint_path):
        print("Descargando pesos pre-entrenados (AudioSet 527 clases, ~300MB)...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print("Descarga completada.")
        
    print("Inyectando pesos pre-entrenados en la red...")
    # Cargar el diccionario de pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # IMPORTANTE: No cargamos los pesos de la última capa (fc_audioset) porque nosotros usamos fc_jamendo
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and k != 'fc_jamendo.weight' and k != 'fc_jamendo.bias'}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print("Transfer Learning completado. El modelo está listo para el Fine-Tuning con MTG-Jamendo.")
    return model

# ── PRUEBA DE FUNCIONAMIENTO ──
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_final = build_panns_model(num_classes=59, device=device)
    
    # Simulamos un batch de 4 audios de 30 segundos muestreados a 32kHz (Onda cruda)
    audio_simulado = torch.randn(4, 32000 * 30).to(device)
    
    print("\nCalculando predicción...")
    with torch.no_grad():
        logits = modelo_final(audio_simulado)
        
    print(f"Forma de salida: {logits.shape} -> (Batch Size, 59 Emociones de Jamendo)")
    print("¡Arquitectura SOTA perfecta!")