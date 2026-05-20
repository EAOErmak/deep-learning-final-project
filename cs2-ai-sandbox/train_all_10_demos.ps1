param (
    [string]$DatasetDir = "dataset_10",
    [int]$ExpectedDemoCount = 10,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 4,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = 0,
    [int]$LogInterval = 10,
    [int]$LogEvery = 10,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 0,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [string]$AimSavePath = "checkpoints\aim_10d_seq16.pt",
    [string]$TrackerSavePath = "checkpoints\enemy_tracker_10d_seq16.pt",
    [string]$MovementSavePath = "checkpoints\movement_10d_seq16.pt"
)

$ErrorActionPreference = 'Stop'

function Invoke-TrainingStep {
    param(
        [string]$Title,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host $Title -ForegroundColor Yellow
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $Title" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$datasetPath = Join-Path $PSScriptRoot $DatasetDir
$cleanPlayPath = Join-Path $datasetPath 'clean_play_ticks'
if (-not (Test-Path $cleanPlayPath)) {
    throw "Dataset directory not found: $cleanPlayPath"
}

$demoFiles = Get-ChildItem $cleanPlayPath -File -Filter *.parquet
$demoCount = @($demoFiles).Count

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Training 3 modules on local dataset" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "DatasetDir=$DatasetDir DemoCount=$demoCount ExpectedDemoCount=$ExpectedDemoCount" -ForegroundColor DarkGray
Write-Host "Epochs=$Epochs BatchSize=$BatchSize SeqLen=$SeqLen Stride=$Stride SplitMode=$SplitMode" -ForegroundColor DarkGray
Write-Host "AimSavePath=$AimSavePath" -ForegroundColor DarkGray
Write-Host "TrackerSavePath=$TrackerSavePath" -ForegroundColor DarkGray
Write-Host "MovementSavePath=$MovementSavePath" -ForegroundColor DarkGray

if ($demoCount -ne $ExpectedDemoCount) {
    Write-Host "Warning: dataset contains $demoCount parquet files, expected $ExpectedDemoCount." -ForegroundColor Red
}

$commonArgs = @{
    DatasetDir = $DatasetDir
    Epochs = $Epochs
    BatchSize = $BatchSize
    SeqLen = $SeqLen
    Stride = $Stride
    SplitMode = $SplitMode
    NumWorkers = $NumWorkers
}
if ($MaxSamples -gt 0) {
    $commonArgs.Add("MaxSamples", $MaxSamples)
}
if ($MaxSamplesPerDemo -gt 0) {
    $commonArgs.Add("MaxSamplesPerDemo", $MaxSamplesPerDemo)
}
if ($ShowIndexProgress) {
    $commonArgs.Add("ShowIndexProgress", $true)
}

Invoke-TrainingStep -Title "[1/4] Training enemy tracker..." -Action {
    $args = @{}
    foreach ($entry in $commonArgs.GetEnumerator()) { $args[$entry.Key] = $entry.Value }
    $args["LogInterval"] = $LogInterval
    $args["SavePath"] = $TrackerSavePath
    & (Join-Path $PSScriptRoot 'train_enemy_tracker.ps1') @args
}

Invoke-TrainingStep -Title "[2/4] Training aim..." -Action {
    $args = @{}
    foreach ($entry in $commonArgs.GetEnumerator()) { $args[$entry.Key] = $entry.Value }
    $args["LogInterval"] = $LogInterval
    $args["SavePath"] = $AimSavePath
    & (Join-Path $PSScriptRoot 'train_aim.ps1') @args
}

Invoke-TrainingStep -Title "[3/4] Training movement..." -Action {
    $args = @{}
    foreach ($entry in $commonArgs.GetEnumerator()) { $args[$entry.Key] = $entry.Value }
    $args["LogEvery"] = $LogEvery
    $args["SavePath"] = $MovementSavePath
    if ($DisableBatchProgress) {
        $args["DisableBatchProgress"] = $true
    }
    & (Join-Path $PSScriptRoot 'train_movement.ps1') @args
}

Invoke-TrainingStep -Title "[4/4] Running train/runtime parity check..." -Action {
    $args = @(
        "scripts/check_train_runtime_feature_parity.py",
        "--dataset-dir", $DatasetDir,
        "--sample-index", 0,
        "--seq-len", $SeqLen,
        "--aim-checkpoint", $AimSavePath,
        "--movement-checkpoint", $MovementSavePath,
        "--tracker-checkpoint", $TrackerSavePath
    )
    python @args
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Training finished successfully." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
