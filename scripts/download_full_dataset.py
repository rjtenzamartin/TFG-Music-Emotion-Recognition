import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

# CONFIGURACIÓN DE RUTAS
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TSV_PATH = PROJECT_ROOT / "mtg-jamendo-dataset" / "data" / "autotagging_moodtheme.tsv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

JAMENDO_BASE = "https://prod-1.storage.jamendo.com/download/track/"
MAX_WORKERS = 4 # Balance perfecto entre velocidad y sigilo

# ── CREACIÓN DE UNA SESIÓN ROBUSTA CON REINTENTOS AUTOMÁTICOS ──
def get_robust_session():
    session = requests.Session()
    # Si da error 429 (bloqueo) o errores de servidor (5xx), reintenta automáticamente
    retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    })
    return session

global_session = get_robust_session()

def download_track(track_info):
    track_id, path_suffix = track_info
    local_path = AUDIO_DIR / path_suffix
    
    # Si ya existe y pesa más de 10KB, saltamos (no perdemos tiempo)
    if local_path.exists() and local_path.stat().st_size > 10000:
        return "exists"
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{JAMENDO_BASE}{track_id}/mp32/"
    
    try:
        response = global_session.get(url, stream=True, timeout=15)
        
        # Si la canción ya no existe en Jamendo (404), la marcamos como "not_found"
        if response.status_code == 404:
            return "not_found"
            
        response.raise_for_status() # Lanza error si no es 200 OK
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
                
        if local_path.stat().st_size > 10000:
            return "downloaded"
        else:
            local_path.unlink()
            return "error_empty"
            
    except Exception as e:
        return f"error_{str(e)[:20]}"

def main():
    if not TSV_PATH.exists():
        print(f"ERROR: No se encuentra {TSV_PATH}")
        return

    tasks = []
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                tasks.append((parts[0], parts[3]))

    print(f"Iniciando descarga robusta de {len(tasks)} canciones...")
    resumen = {"exists": 0, "downloaded": 0, "not_found": 0, "errors": 0}
    
    # Cambiamos a 'as_completed' para procesar los resultados en tiempo real
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Enviamos todas las tareas al pool de trabajadores
        futuros = {executor.submit(download_track, t): t for t in tasks}
        
        # Creamos la barra de progreso
        pbar = tqdm(concurrent.futures.as_completed(futuros), total=len(tasks), unit="song")
        
        try:
            for futuro in pbar:
                track_id, path_suffix = futuros[futuro]
                r = futuro.result() # Obtenemos el resultado de esa canción
                
                # Contabilizamos y escribimos en tiempo real
                if r == "exists":
                    resumen["exists"] += 1
                elif r == "downloaded":
                    resumen["downloaded"] += 1
                    tqdm.write(f"📥 OK - Descargado: {track_id}") 
                elif r == "not_found":
                    resumen["not_found"] += 1
                    tqdm.write(f"👻 404 - Borrado de Jamendo: {track_id}")
                else:
                    resumen["errors"] += 1
                    tqdm.write(f"❌ Error - {track_id}: {r}")
                    
                pbar.set_postfix({
                    'Nuevas': resumen['downloaded'], 
                    'Ya existen': resumen['exists'], 
                    'Borradas': resumen['not_found'], 
                    'Errores': resumen['errors']
                })
                
        except KeyboardInterrupt:
            # ESTO ES LO NUEVO: Parada de emergencia instantánea
            tqdm.write("\n🛑 ¡Parada de emergencia (Ctrl+C) detectada!")
            tqdm.write("Obligando a los hilos a detenerse. Por favor, espera unos segundos...")
            executor.shutdown(wait=False, cancel_futures=True)
            return # Salimos de la función inmediatamente

    print("\n" + "═"*50)
    print("RESULTADO FINAL DE LA DESCARGA")
    print(f"Ya presentes: {resumen['exists']}")
    print(f"Descargadas nuevas: {resumen['downloaded']}")
    print(f"Borradas de Jamendo (404): {resumen['not_found']}")
    print(f"Otros errores: {resumen['errors']}")
    print("═"*50)

if __name__ == "__main__":
    main()