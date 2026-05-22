param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 16,
    [int]$NumWorkers = -1,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [int]$LogEvery = 10,
    [string]$ResumeFrom = '',
    [string]$SavePath = '',
    [string]$Map = 'de_dust2',
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'

Write-Host "Training movement model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'auxiliary\prepare_one_demo_dataset.ps1') -DemoName $DemoName -RequireGridRounds
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\movement_$demoBase.pt"
}
$trainsetDir = Join-Path $datasetDir 'trainsets\movement_solo_grid'
$roundsDatasetGridDir = Join-Path $datasetDir 'rounds-dataset-grid'

$buildArgs = @(
    "-m", "cs2_ai.preprocessing.build_movement_trainset",
    "--rounds-dataset-dir", $roundsDatasetGridDir,
    "--output-dir", $trainsetDir,
    "--map", $Map,
    "--feature-mode", "solo_grid",
    "--split-unit", "round",
    "--require-grid-labels", "true",
    "--seed", $Seed
)

Push-Location $PSScriptRoot
try {
    python @buildArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $args = @(
        "cs2_ai/ml/training/train_movement.py",
        "--trainset-dir", $trainsetDir,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--seq-len", $SeqLen,
        "--stride", $Stride,
        "--split-mode", "round",
        "--num-workers", $NumWorkers,
        "--log-every", $LogEvery,
        "--movement-feature-mode", "solo_grid",
        "--save-path", $SavePath
    )
    if ($ShowIndexProgress) { $args += "--show-index-progress" }
    if ($DisableBatchProgress) { $args += "--disable-batch-progress" }
    if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $args += @("--resume-from", $ResumeFrom) }

    python @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
