param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$EpochsPerRound = 1,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 16,
    [int]$NumWorkers = -1,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [int]$LogEvery = 10,
    [int]$MovementStatsSampleSize = 5000,
    [int]$MaxRounds = 0,
    [switch]$SkipTrainedRounds,
    [switch]$ShuffleRounds,
    [string]$ResumeFrom = '',
    [string]$SavePath = '',
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'

Write-Host "Training movement model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'auxiliary\prepare_one_demo_dataset.ps1') -DemoName $DemoName -RequireGridRounds
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\movement_$demoBase.pt"
}
$roundsDatasetGridDir = Join-Path $datasetDir 'rounds-dataset-grid'

$resolvedRoundsDatasetGridDir = (Resolve-Path -LiteralPath $roundsDatasetGridDir).Path
$datasetSubdir = Split-Path -Path $resolvedRoundsDatasetGridDir -Leaf
$dataDir = Split-Path -Path $resolvedRoundsDatasetGridDir -Parent

Push-Location $PSScriptRoot
try {
    $args = @(
        "cs2_ai/ml/training/train_movement.py",
        "--data-dir", $dataDir,
        "--dataset-subdir", $datasetSubdir,
        "--epochs", $Epochs,
        "--epochs-per-round", $EpochsPerRound,
        "--batch-size", $BatchSize,
        "--seq-len", $SeqLen,
        "--stride", $Stride,
        "--split-mode", "round",
        "--num-workers", $NumWorkers,
        "--log-every", $LogEvery,
        "--movement-feature-mode", "solo_grid",
        "--movement-stats-sample-size", $MovementStatsSampleSize,
        "--save-path", $SavePath,
        "--stream-by-round",
        "--seed", $Seed
    )
    if ($ShowIndexProgress) { $args += "--show-index-progress" }
    if ($DisableBatchProgress) { $args += "--disable-batch-progress" }
    if ($ShuffleRounds) { $args += "--shuffle-rounds" }
    if ($SkipTrainedRounds) { $args += "--skip-trained-rounds" }
    if ($MaxRounds -gt 0) { $args += @("--max-rounds", "$MaxRounds") }
    if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $args += @("--resume-from", $ResumeFrom) }

    python @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
