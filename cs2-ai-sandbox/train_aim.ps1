param (
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 4,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = 4,
    [switch]$ShowIndexProgress,
    [string]$SavePath = "checkpoints\aim_bc_v1.pt"
)

Write-Host "Training aim model..." -ForegroundColor Cyan

$args = @(
    "cs2_ai/ml/training/train_aim.py",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", $SplitMode,
    "--num-workers", $NumWorkers,
    "--save-path", $SavePath
)

if ($ShowIndexProgress) {
    $args += "--show-index-progress"
}

python @args
exit $LASTEXITCODE