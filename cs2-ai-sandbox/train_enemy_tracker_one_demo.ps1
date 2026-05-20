param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 16,
    [int]$NumWorkers = 0,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 20000,
    [switch]$ShowIndexProgress,
    [string]$SavePath = ''
)

Write-Host "Training enemy tracker model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'prepare_one_demo_dataset.ps1') -DemoName $DemoName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\enemy_tracker_$demoBase.pt"
}

$args = @(
    "cs2_ai/ml/training/train_enemy_tracker.py",
    "--dataset-dir", $datasetDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", "round",
    "--num-workers", $NumWorkers,
    "--save-path", $SavePath
)
if ($MaxSamples -gt 0) { $args += @("--max-samples", $MaxSamples) }
if ($MaxSamplesPerDemo -gt 0) { $args += @("--max-samples-per-demo", $MaxSamplesPerDemo) }
if ($ShowIndexProgress) { $args += "--show-index-progress" }

python @args
exit $LASTEXITCODE
