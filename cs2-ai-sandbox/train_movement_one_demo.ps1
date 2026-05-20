param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 16,
    [int]$NumWorkers = 0,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 20000,
    [switch]$ShowIndexProgress,
    [int]$LogEvery = 10,
    [string]$SavePath = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Training movement model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'prepare_one_demo_dataset.ps1') -DemoName $DemoName
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\movement_$demoBase.pt"
}

$args = @(
    "cs2_ai/ml/training/train_movement.py",
    "--dataset-dir", $datasetDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", "round",
    "--num-workers", $NumWorkers,
    "--log-every", $LogEvery,
    "--save-path", $SavePath
)
if ($MaxSamples -gt 0) { $args += @("--max-samples", $MaxSamples) }
if ($MaxSamplesPerDemo -gt 0) { $args += @("--max-samples-per-demo", $MaxSamplesPerDemo) }
if ($ShowIndexProgress) { $args += "--show-index-progress" }

python @args
exit $LASTEXITCODE
