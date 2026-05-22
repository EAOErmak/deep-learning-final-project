param (
    [string]$RoundsDatasetDir = "data\processed\rounds-dataset-grid",
    [string]$OutputTrainsetDir = "data\trainsets\movement_solo_grid",
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 8,
    [string]$SplitMode = "round",
    [int]$NumWorkers = -1,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [int]$LogEvery = 10,
    [string]$SavePath = "checkpoints\movement_bc_v1.pt",
    [string]$Map = "de_dust2",
    [int]$Seed = 42
)

Write-Host "Building movement trainset from rounds dataset..." -ForegroundColor Cyan
Write-Host "RoundsDatasetDir=$RoundsDatasetDir OutputTrainsetDir=$OutputTrainsetDir Map=$Map SplitMode=$SplitMode" -ForegroundColor DarkGray

$buildArgs = @(
    "-m", "cs2_ai.preprocessing.build_movement_trainset",
    "--rounds-dataset-dir", $RoundsDatasetDir,
    "--output-dir", $OutputTrainsetDir,
    "--map", $Map,
    "--feature-mode", "solo_grid",
    "--split-unit", $SplitMode,
    "--require-grid-labels", "true",
    "--seed", $Seed
)

python @buildArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Training movement model from prebuilt trainset..." -ForegroundColor Cyan

$args = @(
    "cs2_ai/ml/training/train_movement.py",
    "--trainset-dir", $OutputTrainsetDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", $SplitMode,
    "--num-workers", $NumWorkers,
    "--log-every", $LogEvery,
    "--movement-feature-mode", "solo_grid",
    "--save-path", $SavePath
)

if ($ShowIndexProgress) {
    $args += "--show-index-progress"
}

if ($DisableBatchProgress) {
    $args += "--disable-batch-progress"
}

python @args
exit $LASTEXITCODE
