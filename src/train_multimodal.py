"""
train_multimodal.py — Fusión Multimodal Temporal (Late Fusion)
==============================================================
TFG: Reconocimiento de Emociones Musicales

Modalidades:
1. Visual:    AST     (30 frames x 768  dims)
2. Acústica:  PANNs   (30 frames x 2048 dims)
3. Semántica: Whisper (30 frames x 512  dims)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path
import random

def set_seed(seed: int = 42):
    print(f"Fijando semilla de reproducibilidad a: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

# LLAMAR A LA FUNCIÓN INMEDIATAMENTE
set_seed(42)

# ── 1. CONFIGURACIÓN ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Al ser features pre-calculadas, podemos usar un batch grande
EPOCHS = 30
LR = 1e-3

# CORRECCIÓN: Path relativo al script, no al CWD (robusto al directorio de lanzamiento)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT / 'data' / 'embeddings_fase3'

# ── SPLITS OFICIALES SPLIT-0 (Anti Data-Leakage / Efecto Artista) ──
SPLIT_DIR = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
TRAIN_TSV = SPLIT_DIR / "autotagging_moodtheme-train.tsv"
VAL_TSV   = SPLIT_DIR / "autotagging_moodtheme-validation.tsv"
TEST_TSV  = SPLIT_DIR / "autotagging_moodtheme-test.tsv"

print(f"Iniciando Entrenamiento Multimodal en {DEVICE}")


# ── 2. DATASET MULTIMODAL ROBUSTO ──
class MultimodalJamendoDataset(Dataset):
    """Dataset de embeddings pre-extraídos para el modelo Late Fusion.

    Parameters
    ----------
    tsv_path : Path
        Ruta a UNO de los TSVs oficiales (train, validation o test).
        NUNCA pasar el TSV completo + hacer sample() aquí dentro.
    embeddings_dir : Path
        Carpeta con las matrices .npy de AST, PANNs y Whisper.
    label_list : list[str] | None
        Lista ordenada de las 59 etiquetas. Si se pasa (obligatorio para val/test),
        se usa directamente. Si es None, se deriva de los datos del propio TSV
        (solo para train). Garantiza que val y test usan el mismo orden que train.
    """
    def __init__(self, tsv_path: Path, embeddings_dir: Path, label_list: list[str] | None = None):
        print(f"Cargando TSV: {tsv_path.name}")
        registros = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            next(f)  # Saltar cabecera: TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS
            for linea in f:
                if not linea.strip():
                    continue
                parts = linea.strip().split('\t')
                if len(parts) >= 6:
                    # Extraer ID numérico del track (ej. "track_0000948" → "948")
                    track_id = parts[0].replace('track_', '').lstrip('0')
                    if track_id == '':
                        track_id = '0'
                    # Filtrar etiquetas mood/theme y limpiar prefijo
                    tags = [t.replace('mood/theme---', '').strip()
                            for t in parts[5:] if t.startswith('mood/theme---')]
                    if tags:
                        registros.append({
                            'track_id': track_id,
                            'tags': ','.join(tags)
                        })

        self.df = pd.DataFrame(registros)

        # Vocabulario de etiquetas:
        # - Train: derivado del TSV completo del benchmark (59 etiquetas fijas)
        #   Las 3 etiquetas que faltan en los splits existen en autotagging_moodtheme.tsv
        #   pero sus tracks fueron excluidos del split-0. La union de splits da 56, no 59.
        # - Val/Test: reciben el mismo label_list que uso train (CRITICO para alinear tensores)
        if label_list is None:
            _full_tsv = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
            _todas: set = set()
            with open(_full_tsv, "r", encoding="utf-8") as _fh:
                next(_fh)
                for _ln in _fh:
                    _parts = _ln.strip().split("\t")
                    if len(_parts) >= 6:
                        _todas.update(
                            t.replace("mood/theme---", "").strip()
                            for t in _parts[5:]
                            if t.startswith("mood/theme---")
                        )
            self.etiquetas_jamendo = sorted(_todas)  # -> 59 etiquetas
        else:
            self.etiquetas_jamendo = label_list

        self.label2idx = {label: i for i, label in enumerate(self.etiquetas_jamendo)}
        print(f"  >>{len(self.df)} canciones | {len(self.etiquetas_jamendo)} etiquetas")

        # Cargar matrices de embeddings pre-extraídas
        print(f"  Cargando matrices .npy de embeddings...")
        embeddings_dir = Path(embeddings_dir)
        ast_embs    = np.load(embeddings_dir / 'ast_temporal_embeddings.npy')
        ast_ids     = np.load(embeddings_dir / 'track_ids_ast_temporal.npy')
        panns_embs  = np.load(embeddings_dir / 'panns_temporal_embeddings.npy')
        panns_ids   = np.load(embeddings_dir / 'track_ids_panns_temporal.npy')
        whisper_embs = np.load(embeddings_dir / 'whisper_temporal_embeddings.npy')
        whisper_ids  = np.load(embeddings_dir / 'track_ids_whisper_temporal.npy')

        # Diccionarios track_id → embedding (robusto ante fallos parciales de extracción)
        self.ast_dict     = {str(tid): emb for tid, emb in zip(ast_ids,     ast_embs)}
        self.panns_dict   = {str(tid): emb for tid, emb in zip(panns_ids,   panns_embs)}
        self.whisper_dict = {str(tid): emb for tid, emb in zip(whisper_ids, whisper_embs)}

        # Filtrar solo canciones con las 3 modalidades extraídas correctamente
        ids_validos = (
            set(self.ast_dict.keys()) &
            set(self.panns_dict.keys()) &
            set(self.whisper_dict.keys())
        )
        self.df = self.df[self.df['track_id'].isin(ids_validos)].reset_index(drop=True)
        print(f"  Canciones válidas con las 3 modalidades: {len(self.df)}")

    def get_pos_weight(self) -> torch.Tensor:
        """Calcula pos_weight para BCEWithLogitsLoss (corrección de desbalanceo).

        CORRECCIÓN: El modelo original no usaba pos_weight, lo que penalizaba
        fuertemente las etiquetas raras. Ahora se corrige con la razón neg/pos.
        """
        n_total = len(self.df)
        n_classes = len(self.etiquetas_jamendo)
        pos_counts = np.zeros(n_classes, dtype=np.float64)

        for tags_str in self.df['tags']:
            for tag in tags_str.split(','):
                if tag in self.label2idx:
                    pos_counts[self.label2idx[tag]] += 1

        eps = 1e-6
        pos_counts = np.maximum(pos_counts, eps)
        neg_counts = n_total - pos_counts
        weight = neg_counts / pos_counts
        return torch.tensor(weight, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tid = row['track_id']   # CORRECCIÓN: la columna se llama 'track_id', no 'track_id_clean'
        
        # Recuperar secuencias temporales (30 frames)
        ast_seq = torch.tensor(self.ast_dict[tid], dtype=torch.float32)
        panns_seq = torch.tensor(self.panns_dict[tid], dtype=torch.float32)
        whisper_seq = torch.tensor(self.whisper_dict[tid], dtype=torch.float32)
        
        # One-Hot Encoding para Multi-Label (59 clases)
        target = torch.zeros(len(self.etiquetas_jamendo), dtype=torch.float32)
        if pd.notna(row['tags']):
            tags = row['tags'].split(',')
            for tag in tags:
                if tag in self.label2idx:
                    target[self.label2idx[tag]] = 1.0
                    
        return ast_seq, panns_seq, whisper_seq, target

# ── 3. ARQUITECTURA: LATE FUSION MLP ──
class MultimodalLateFusion(nn.Module):
    def __init__(self, num_classes=59):
        super().__init__()
        
        # Las dimensiones de las 3 ramas concatenadas
        # AST (768) + PANNs (2048) + Whisper (512) = 3328
        in_features = 768 + 2048 + 512
        
        # El MLP Perceptrón Multicapa (El "Cerebro")
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Capa final de salida para las 59 etiquetas
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ast, panns, whisper):
        # 1. Global Average Pooling (Temporal)
        # Transformamos (Batch, 30, Dims) -> (Batch, Dims)
        # Promediamos la emoción a lo largo de los 30 segundos
        ast_pooled = ast.mean(dim=1)
        panns_pooled = panns.mean(dim=1)
        whisper_pooled = whisper.mean(dim=1)
        
        # 2. Late Fusion (Concatenación en el espacio latente)
        fused_features = torch.cat([ast_pooled, panns_pooled, whisper_pooled], dim=1)
        
        # 3. Clasificación
        logits = self.mlp(fused_features)
        return logits

# ── 4. BUCLE DE ENTRENAMIENTO ──
def main():
    # Cargar train primero para derivar el vocabulario de etiquetas
    train_ds = MultimodalJamendoDataset(TRAIN_TSV, EMBEDDINGS_DIR, label_list=None)
    # Pasar el label_list de train a val para garantizar el mismo orden de clases
    val_ds = MultimodalJamendoDataset(VAL_TSV, EMBEDDINGS_DIR, label_list=train_ds.etiquetas_jamendo)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultimodalLateFusion(num_classes=59).to(DEVICE)

    # CORRECCIÓN: Añadir pos_weight para corregir el desbalanceo masivo de clases.
    # Sin esto, el modelo ignora las etiquetas raras (aprende a predecir siempre 0).
    pos_weights = train_ds.get_pos_weight().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    best_val_auc = 0.0
    history = []  # Para guardar curvas de aprendizaje (requerido por el notebook de análisis)

    for epoch in range(1, EPOCHS + 1):
        # ── TRAIN ──
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for ast_x, panns_x, whisper_x, y in pbar:
            ast_x, panns_x, whisper_x, y = (
                ast_x.to(DEVICE), panns_x.to(DEVICE), whisper_x.to(DEVICE), y.to(DEVICE)
            )
            optimizer.zero_grad(set_to_none=True)   # set_to_none=True libera VRAM explícitamente
            logits = model(ast_x, panns_x, whisper_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # ── VAL ──
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for ast_x, panns_x, whisper_x, y in val_loader:
                ast_x, panns_x, whisper_x, y = (
                    ast_x.to(DEVICE), panns_x.to(DEVICE), whisper_x.to(DEVICE), y.to(DEVICE)
                )
                logits = model(ast_x, panns_x, whisper_x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(y.cpu().numpy())

        y_true  = np.vstack(all_targets)
        y_score = np.vstack(all_preds)

        valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
        val_auc = roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro')

        scheduler.step(val_auc)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print(f"Resumen Epoca {epoch}: Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | ROC-AUC: {val_auc:.4f}")

        # Guardar historial de métricas por época (necesario para los notebooks de análisis)
        history.append({
            'epoch':      epoch,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
            'val_auc':    val_auc,
        })
        pd.DataFrame(history).to_csv(SCRIPT_DIR / "history_multimodal.csv", index=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SCRIPT_DIR / "best_multimodal_model.pth")
            print(f"  >>Nuevo mejor modelo guardado (AUC: {best_val_auc:.4f})")

    # ── GRÁFICAS DE CURVAS DE APRENDIZAJE (PNG para la memoria) ─────────────
    _save_learning_curves(
        csv_path   = SCRIPT_DIR / "history_multimodal.csv",
        out_path   = SCRIPT_DIR / "multimodal_learning_curves.png",
        model_name = "Late Fusion Multimodal",
    )
    print(f"\nEntrenamiento Multimodal finalizado. Mejor AUC validación: {best_val_auc:.4f}")


def _save_learning_curves(csv_path, out_path, model_name: str) -> None:
    """Genera y guarda las curvas de Loss y ROC-AUC como PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[AVISO] No se pudo leer {csv_path}: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Curvas de Aprendizaje", fontsize=14, fontweight="bold")

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#ff7f0e", lw=2, ls="--")
    ax1.set_xlabel("Época"); ax1.set_ylabel("BCEWithLogitsLoss")
    ax1.set_title("Función de Pérdida", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["val_auc"], color="#2ca02c", lw=2,
             marker="o", markersize=4, label="Val ROC-AUC")
    best_row = df.loc[df["val_auc"].idxmax()]
    ax2.axhline(best_row["val_auc"], color="red", ls="--", alpha=0.5,
                label=f"Máx {best_row['val_auc']:.4f} (ep.{int(best_row['epoch'])})")
    ax2.axhline(0.725, color="gray", ls=":", alpha=0.7, label="Baseline 0.725")
    ax2.set_xlabel("Época"); ax2.set_ylabel("ROC-AUC Macro")
    ax2.set_title("ROC-AUC en Validación", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] Curvas de aprendizaje guardadas: {out_path}")


if __name__ == "__main__":
    main()
