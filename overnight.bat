@echo off
chcp 65001 > nul
echo [%date% %time%] === INICIO ENTRENAMIENTO OVERNIGHT === >> overnight.log
echo [%date% %time%] === INICIO ENTRENAMIENTO OVERNIGHT ===

call conda activate gpu_env

echo [%date% %time%] Paso 1/5: train_panns.py >> overnight.log
echo Paso 1/5: Entrenando PANNs CNN14...
python src\train_panns.py >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en train_panns.py - Abortando >> overnight.log
    echo FALLO en train_panns.py
    goto :fin
)
echo [%date% %time%] Paso 1 COMPLETADO >> overnight.log

echo [%date% %time%] Paso 2/5: train_ast.py >> overnight.log
echo Paso 2/5: Entrenando AST Transformer...
python src\train_ast.py >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en train_ast.py - Abortando >> overnight.log
    echo FALLO en train_ast.py
    goto :fin
)
echo [%date% %time%] Paso 2 COMPLETADO >> overnight.log

echo [%date% %time%] Paso 3/5: train_multimodal.py >> overnight.log
echo Paso 3/5: Entrenando Late Fusion Multimodal...
python src\train_multimodal.py >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en train_multimodal.py - Abortando >> overnight.log
    echo FALLO en train_multimodal.py
    goto :fin
)
echo [%date% %time%] Paso 3 COMPLETADO >> overnight.log

echo [%date% %time%] Paso 4/5: evaluacion_final.py >> overnight.log
echo Paso 4/5: Evaluacion final (FASE 4, 5, 6a, tabla)...
python src\evaluacion_final.py --fase 4 5 6a tabla >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en evaluacion_final.py - Abortando >> overnight.log
    echo FALLO en evaluacion_final.py
    goto :fin
)
echo [%date% %time%] Paso 4 COMPLETADO >> overnight.log

echo [%date% %time%] Paso 5/5: fase7_clap_reeval.py >> overnight.log
echo Paso 5/5: Evaluacion CLAP Zero-Shot corregida...
python src\fase7_clap_reeval.py >> overnight.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] FALLO en fase7_clap_reeval.py >> overnight.log
    echo FALLO en fase7_clap_reeval.py
    goto :fin
)
echo [%date% %time%] Paso 5 COMPLETADO >> overnight.log

:fin
echo [%date% %time%] === FIN DEL PROCESO === >> overnight.log
echo === TODO COMPLETADO ===
