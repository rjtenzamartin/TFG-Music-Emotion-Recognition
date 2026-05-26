@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

:: Activar entorno conda con GPU
call "C:\Users\Usuario\miniconda3\Scripts\activate.bat" gpu_env

cd /d "C:\Users\Usuario\OneDrive - UNIVERSIDAD DE MURCIA\Escritorio\CIENCIA DE DATOS RUBEN\CUARTO\TFG_Emociones"

echo [%date% %time%] === INICIO EVALUACION FINAL === >> overnight.log

echo [%date% %time%] Iniciando evaluacion_final.py >> overnight.log
python src\evaluacion_final.py --fase 5 6a tabla >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en evaluacion_final.py - Abortando >> overnight.log
    goto :fin
)
echo [%date% %time%] evaluacion_final.py COMPLETADO >> overnight.log

echo [%date% %time%] Iniciando fase7_clap_reeval.py >> overnight.log
python src\fase7_clap_reeval.py >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en fase7_clap_reeval.py >> overnight.log
    goto :fin
)
echo [%date% %time%] fase7_clap_reeval.py COMPLETADO >> overnight.log

:fin
echo [%date% %time%] === FIN DEL PROCESO === >> overnight.log
