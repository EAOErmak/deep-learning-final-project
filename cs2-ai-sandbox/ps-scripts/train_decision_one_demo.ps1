param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [int]$SeqLen = 64,
    [int]$Stride = 16,
    [int]$NumWorkers = -1,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 20000,
    [int]$MaxCachedDemos = 2,
    [int]$LogInterval = 10,
    [string]$ResumeFrom = '',
    [string]$SavePath = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Training decision model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'prepare_one_demo_dataset.ps1') -DemoName $DemoName
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\decision_$demoBase.pt"
}

$args = @(
    "cs2_ai/ml/training/train_decision_offline.py",
    "--dataset-dir", $datasetDir,
    "--dataset-subdir", "rounds-dataset",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", "round",
    "--num-workers", $NumWorkers,
    "--max-cached-demos", $MaxCachedDemos,
    "--log-interval", $LogInterval,
    "--save-path", $SavePath
)
if ($MaxSamples -gt 0) { $args += @("--max-samples", $MaxSamples) }
if ($MaxSamplesPerDemo -gt 0) { $args += @("--max-samples-per-demo", $MaxSamplesPerDemo) }
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $args += @("--resume-from", $ResumeFrom) }

Push-Location $PSScriptRoot
try {
    python @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
