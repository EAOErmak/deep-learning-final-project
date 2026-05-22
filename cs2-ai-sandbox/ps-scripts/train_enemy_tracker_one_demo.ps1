param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 16,
    [int]$NumWorkers = -1,
    [int]$LogInterval = 10,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 20000,
    [switch]$ShowIndexProgress,
    [switch]$StreamByRound,
    [int]$EpochsPerRound = 1,
    [switch]$ShuffleRounds,
    [int]$MaxRounds = 0,
    [string]$ResumeFrom = '',
    [string]$SavePath = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Training enemy tracker model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'auxiliary\prepare_one_demo_dataset.ps1') -DemoName $DemoName
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\enemy_tracker_$demoBase.pt"
}

$args = @(
    "cs2_ai/ml/training/train_enemy_tracker.py",
    "--dataset-dir", $datasetDir,
    "--dataset-subdir", "rounds-dataset",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", "round",
    "--num-workers", $NumWorkers,
    "--log-interval", $LogInterval,
    "--save-path", $SavePath
)
if ($MaxSamples -gt 0) { $args += @("--max-samples", $MaxSamples) }
if ($MaxSamplesPerDemo -gt 0) { $args += @("--max-samples-per-demo", $MaxSamplesPerDemo) }
if ($ShowIndexProgress) { $args += "--show-index-progress" }
if ($StreamByRound) {
    $args += @("--stream-by-round", "--epochs-per-round", $EpochsPerRound)
    if ($ShuffleRounds) { $args += "--shuffle-rounds" }
    if ($MaxRounds -gt 0) { $args += @("--max-rounds", $MaxRounds) }
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $args += @("--resume-from", $ResumeFrom) }

Push-Location $PSScriptRoot
try {
    python @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
