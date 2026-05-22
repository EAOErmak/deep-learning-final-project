param (
    [string]$DataDir = "data\processed",
    [string]$DatasetSubdir = "rounds-dataset",
    [int]$EpochsPerDemo = 1,
    [int]$TrackerBatchSize = 64,
    [int]$AimBatchSize = 64,
    [int]$MovementBatchSize = 64,
    [int]$DecisionBatchSize = 32,
    [int]$TrackerSeqLen = 16,
    [int]$AimSeqLen = 16,
    [int]$MovementSeqLen = 64,
    [int]$DecisionSeqLen = 64,
    [int]$TrackerStride = 16,
    [int]$AimStride = 16,
    [int]$MovementStride = 16,
    [int]$DecisionStride = 16,
    [int]$NumWorkers = -1,
    [int]$LogInterval = 10,
    [int]$LogEvery = 10,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 20000,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [string]$TrackerSavePath = "checkpoints\enemy_tracker_stream.pt",
    [string]$AimSavePath = "checkpoints\aim_stream.pt",
    [string]$MovementSavePath = "checkpoints\movement_stream.pt",
    [string]$DecisionSavePath = "checkpoints\decision_stream.pt"
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {

function Invoke-Step {
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

function Resolve-DemoBaseName {
    param([Parameter(Mandatory = $true)][string]$Name)

    $base = [System.IO.Path]::GetFileName($Name)
    foreach ($suffix in @('_play_ticks.parquet', '_buy_ticks.parquet', '.parquet', '.dem')) {
        if ($base.EndsWith($suffix, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $base.Substring(0, $base.Length - $suffix.Length)
        }
    }
    return $base
}

$datasetPath = Join-Path $PSScriptRoot $DataDir
$roundsDatasetPath = Join-Path $datasetPath $DatasetSubdir
if (-not (Test-Path -LiteralPath $roundsDatasetPath)) {
    throw "Dataset directory not found: $roundsDatasetPath"
}

$demoDirs = Get-ChildItem -Path $roundsDatasetPath -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'rounds') } | Sort-Object Name
if (-not $demoDirs) {
    throw "No round demo directories found in: $roundsDatasetPath"
}

$demoNames = @($demoDirs | ForEach-Object { Resolve-DemoBaseName -Name $_.Name })

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Streaming training: 1 demo at a time" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "DataDir=$DataDir DatasetSubdir=$DatasetSubdir DemoCount=$($demoNames.Count) EpochsPerDemo=$EpochsPerDemo" -ForegroundColor DarkGray
Write-Host "TrackerSavePath=$TrackerSavePath" -ForegroundColor DarkGray
Write-Host "AimSavePath=$AimSavePath" -ForegroundColor DarkGray
Write-Host "MovementSavePath=$MovementSavePath" -ForegroundColor DarkGray
Write-Host "DecisionSavePath=$DecisionSavePath" -ForegroundColor DarkGray

$demoIndex = 0
foreach ($demoName in $demoNames) {
    $demoIndex += 1
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "Demo $demoIndex/$($demoNames.Count): $demoName" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan

    Invoke-Step -Title "[1/4] Enemy tracker | $demoName" -Action {
        $args = @{
            DemoName = $demoName
            Epochs = $EpochsPerDemo
            BatchSize = $TrackerBatchSize
            SeqLen = $TrackerSeqLen
            Stride = $TrackerStride
            NumWorkers = $NumWorkers
            LogInterval = $LogInterval
            SavePath = $TrackerSavePath
        }
        if (Test-Path -LiteralPath $TrackerSavePath) { $args["ResumeFrom"] = $TrackerSavePath }
        if ($MaxSamples -gt 0) { $args["MaxSamples"] = $MaxSamples }
        if ($MaxSamplesPerDemo -gt 0) { $args["MaxSamplesPerDemo"] = $MaxSamplesPerDemo }
        if ($ShowIndexProgress) { $args["ShowIndexProgress"] = $true }
        & (Join-Path $PSScriptRoot 'train_enemy_tracker_one_demo.ps1') @args
    }

    Invoke-Step -Title "[2/4] Aim | $demoName" -Action {
        $args = @{
            DemoName = $demoName
            Epochs = $EpochsPerDemo
            BatchSize = $AimBatchSize
            SeqLen = $AimSeqLen
            Stride = $AimStride
            NumWorkers = $NumWorkers
            LogInterval = $LogInterval
            SavePath = $AimSavePath
        }
        if (Test-Path -LiteralPath $AimSavePath) { $args["ResumeFrom"] = $AimSavePath }
        if ($MaxSamples -gt 0) { $args["MaxSamples"] = $MaxSamples }
        if ($MaxSamplesPerDemo -gt 0) { $args["MaxSamplesPerDemo"] = $MaxSamplesPerDemo }
        if ($ShowIndexProgress) { $args["ShowIndexProgress"] = $true }
        & (Join-Path $PSScriptRoot 'train_aim_one_demo.ps1') @args
    }

    Invoke-Step -Title "[3/4] Movement | $demoName" -Action {
        $args = @{
            DemoName = $demoName
            Epochs = $EpochsPerDemo
            BatchSize = $MovementBatchSize
            SeqLen = $MovementSeqLen
            Stride = $MovementStride
            NumWorkers = $NumWorkers
            LogEvery = $LogEvery
            SavePath = $MovementSavePath
        }
        if (Test-Path -LiteralPath $MovementSavePath) { $args["ResumeFrom"] = $MovementSavePath }
        if ($MaxSamples -gt 0) { $args["MaxSamples"] = $MaxSamples }
        if ($MaxSamplesPerDemo -gt 0) { $args["MaxSamplesPerDemo"] = $MaxSamplesPerDemo }
        if ($ShowIndexProgress) { $args["ShowIndexProgress"] = $true }
        if ($DisableBatchProgress) { $args["DisableBatchProgress"] = $true }
        & (Join-Path $PSScriptRoot 'train_movement_one_demo.ps1') @args
    }

    Invoke-Step -Title "[4/4] Decision | $demoName" -Action {
        $args = @{
            DemoName = $demoName
            Epochs = $EpochsPerDemo
            BatchSize = $DecisionBatchSize
            SeqLen = $DecisionSeqLen
            Stride = $DecisionStride
            NumWorkers = $NumWorkers
            LogInterval = $LogInterval
            SavePath = $DecisionSavePath
        }
        if (Test-Path -LiteralPath $DecisionSavePath) { $args["ResumeFrom"] = $DecisionSavePath }
        if ($MaxSamples -gt 0) { $args["MaxSamples"] = $MaxSamples }
        if ($MaxSamplesPerDemo -gt 0) { $args["MaxSamplesPerDemo"] = $MaxSamplesPerDemo }
        & (Join-Path $PSScriptRoot 'train_decision_one_demo.ps1') @args
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Streaming one-demo-at-a-time training finished." -ForegroundColor Green
Write-Host "TensorBoard: tensorboard --logdir=runs" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Green
} finally {
    Pop-Location
}
