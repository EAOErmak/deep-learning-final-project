param (
    [string]$DataDir = "data\processed",
    [string]$DatasetSubdir = "rounds-dataset",
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 4,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = -1,
    [int]$LogInterval = 10,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 0,
    [switch]$ShowIndexProgress,
    [string]$AimFeatureMode = "vision_like",
    [string]$SavePath = "checkpoints\aim_bc_v1.pt"
)

Write-Host "Training aim model..." -ForegroundColor Cyan
Write-Host "DataDir=$DataDir DatasetSubdir=$DatasetSubdir Epochs=$Epochs BatchSize=$BatchSize SeqLen=$SeqLen Stride=$Stride SplitMode=$SplitMode" -ForegroundColor DarkGray

$args = @(
    "cs2_ai/ml/training/train_aim.py",
    "--data-dir", $DataDir,
    "--dataset-subdir", $DatasetSubdir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", $SplitMode,
    "--num-workers", $NumWorkers,
    "--log-interval", $LogInterval,
    "--aim-feature-mode", $AimFeatureMode,
    "--save-path", $SavePath
)

if ($MaxSamples -gt 0) {
    $args += "--max-samples"
    $args += $MaxSamples
}

if ($MaxSamplesPerDemo -gt 0) {
    $args += "--max-samples-per-demo"
    $args += $MaxSamplesPerDemo
}

if ($ShowIndexProgress) {
    $args += "--show-index-progress"
}

python @args
exit $LASTEXITCODE
