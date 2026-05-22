$PROJECT_DIR = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_DIR

# Попытка использовать python из виртуального окружения, если оно есть
$PYTHON_EXE = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PYTHON_EXE = ".venv\Scripts\python.exe"
}

Write-Host "Starting Realtime 3D Viewer..." -ForegroundColor Cyan

& $PYTHON_EXE scripts/visualize_rounds_grid_future_cells_3d.py `
    --input "data\processed\rounds-dataset-grid\9z-vs-alka-m1-dust2_play_ticks\rounds\round_0.parquet" `
    --ticks-per-second 50 `
    --tick-step 1 `
    --future-cells 50 `
    --grid-size 700 `
    --grid-offset-x -225 `
    --grid-offset-y 975 `
    --grid-offset-z 0 `
    --bg-image "de_dust2_radar.png" `
    --bg-scale 4.635
