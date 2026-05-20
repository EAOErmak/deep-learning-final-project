param (
    [int]$epochs = 20,
    [int]$batchSize = 32
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Full Neural AI Pipeline Training" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Train Enemy Tracker (Seq2Seq)
Write-Host "`n[1/3] Training Enemy Tracker..." -ForegroundColor Yellow
python cs2_ai/ml/training/train_enemy_tracker.py --epochs $epochs --batch-size $batchSize
if ($LASTEXITCODE -ne 0) {
    Write-Host "Enemy Tracker training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Train Movement (Seq2Seq)
Write-Host "`n[2/3] Training Movement Model..." -ForegroundColor Yellow
python cs2_ai/ml/training/train_movement.py --epochs $epochs --batch-size $batchSize
if ($LASTEXITCODE -ne 0) {
    Write-Host "Movement training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Train Aim (W2S Vision)
Write-Host "`n[3/3] Training Aim Model (W2S 2D Vision)..." -ForegroundColor Yellow
python cs2_ai/ml/training/train_aim.py --epochs $epochs --batch-size $batchSize
if ($LASTEXITCODE -ne 0) {
    Write-Host "Aim training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "All models trained successfully!" -ForegroundColor Green
Write-Host "You can view the metrics in Tensorboard by running:" -ForegroundColor Green
Write-Host "tensorboard --logdir=runs" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Green
