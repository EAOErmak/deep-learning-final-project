# Examples:
# .\ps-scripts\train_aim_one_demo.ps1 -DemoName demo_name -MaxRounds 10
# .\ps-scripts\train_aim_one_demo.ps1 -DemoName demo_name -MaxRounds 10 -SkipTrainedRounds
# .\ps-scripts\train_aim_one_demo.ps1 -DemoName demo_name

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
    [int]$MaxRounds = 0,
    [switch]$SkipTrainedRounds,
    [switch]$ShowIndexProgress,
    [string]$ResumeFrom = '',
    [string]$SavePath = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Training aim model on one demo..." -ForegroundColor Cyan
if ($MaxRounds -gt 0) {
    Write-Host "AIM training max rounds: $MaxRounds" -ForegroundColor DarkGray
} else {
    Write-Host "AIM training max rounds: unlimited" -ForegroundColor DarkGray
}
Write-Host "AIM skip trained rounds: $([bool]$SkipTrainedRounds)" -ForegroundColor DarkGray
$datasetDir = & (Join-Path $PSScriptRoot 'auxiliary\prepare_one_demo_dataset.ps1') -DemoName $DemoName
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\aim_$demoBase.pt"
}

$args = @(
    "cs2_ai/ml/training/train_aim.py",
    "--dataset-dir", $datasetDir,
    "--dataset-subdir", "rounds-dataset",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", "round",
    "--num-workers", $NumWorkers,
    "--log-interval", $LogInterval,
    "--aim-feature-mode", "vision_like",
    "--save-path", $SavePath
)
if ($MaxSamples -gt 0) { $args += @("--max-samples", $MaxSamples) }
if ($MaxSamplesPerDemo -gt 0) { $args += @("--max-samples-per-demo", $MaxSamplesPerDemo) }
if ($ShowIndexProgress) { $args += "--show-index-progress" }
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $args += @("--resume-from", $ResumeFrom) }
if ($SkipTrainedRounds) { $args += "--skip-trained-rounds" }
if ($MaxRounds -gt 0) {
    $args += @(
        "--stream-by-round",
        "--epochs-per-round", "1",
        "--binary-stats-sample-size", "5000",
        "--max-rounds", "$MaxRounds"
    )
}

Push-Location $PSScriptRoot
try {
    python @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
