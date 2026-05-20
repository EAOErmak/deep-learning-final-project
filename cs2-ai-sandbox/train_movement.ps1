param (
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 8,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = 4,
    [switch]$ShowIndexProgress,
    [int]$LogEvery = 10,
    [string]$SavePath = "checkpoints\movement_bc_v1.pt"
)

Write-Host "Training movement model..." -ForegroundColor Cyan

$args = @(
    "cs2_ai/ml/training/train_movement.py",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", $SplitMode,
    "--num-workers", $NumWorkers,
    "--log-every", $LogEvery,
    "--save-path", $SavePath
)

if ($ShowIndexProgress) {
    $args += "--show-index-progress"
}

python @args
exit $LASTEXITCODE