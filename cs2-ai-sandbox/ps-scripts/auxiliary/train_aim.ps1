# Examples:
# .\ps-scripts\auxiliary\train_aim.ps1 -MaxRounds 50
# .\ps-scripts\auxiliary\train_aim.ps1 -MaxRounds 100 -SkipTrainedRounds
# .\ps-scripts\auxiliary\train_aim.ps1

param (
    [string]$DataDir = "data\processed",
    [string]$DatasetSubdir = "rounds-dataset",

    [int]$BatchSize = 64,
    [int]$SeqLen = 16,
    [int]$Stride = 4,
    [int]$NumWorkers = -1,
    [int]$LogInterval = 10,

    [int]$MaxRounds = 0,
    [int]$EpochsPerRound = 1,
    [int]$BinaryStatsSampleSize = 5000,

    [switch]$SkipTrainedRounds,
    [switch]$ShowIndexProgress,

    [string]$AimFeatureMode = "vision_like",
    [string]$AimHeadMode = "legacy",
    [string]$SavePath = "checkpoints\aim_bc_v1.pt"
)

Write-Host "Training AIM model..." -ForegroundColor Cyan
Write-Host "DataDir=$DataDir DatasetSubdir=$DatasetSubdir BatchSize=$BatchSize SeqLen=$SeqLen Stride=$Stride" -ForegroundColor DarkGray
Write-Host "AIM feature mode: $AimFeatureMode" -ForegroundColor DarkGray
Write-Host "AIM head mode: $AimHeadMode" -ForegroundColor DarkGray
Write-Host "AIM stream-by-round: true" -ForegroundColor DarkGray
Write-Host "AIM shuffle rounds: true" -ForegroundColor DarkGray
Write-Host "AIM epochs per round: $EpochsPerRound" -ForegroundColor DarkGray
Write-Host "AIM binary stats sample size: $BinaryStatsSampleSize" -ForegroundColor DarkGray

if ($MaxRounds -gt 0) {
    Write-Host "AIM training max rounds: $MaxRounds" -ForegroundColor DarkGray
} else {
    Write-Host "AIM training max rounds: unlimited" -ForegroundColor DarkGray
}

Write-Host "AIM skip trained rounds: $([bool]$SkipTrainedRounds)" -ForegroundColor DarkGray
Write-Host "SavePath=$SavePath" -ForegroundColor DarkGray

$TrainArgs = @(
    "cs2_ai/ml/training/train_aim.py",
    "--data-dir", $DataDir,
    "--dataset-subdir", $DatasetSubdir,
    "--batch-size", "$BatchSize",
    "--seq-len", "$SeqLen",
    "--stride", "$Stride",
    "--num-workers", "$NumWorkers",
    "--log-interval", "$LogInterval",
    "--aim-feature-mode", $AimFeatureMode,
    "--aim-head-mode", $AimHeadMode,
    "--save-path", $SavePath,
    "--stream-by-round",
    "--shuffle-rounds",
    "--epochs-per-round", "$EpochsPerRound",
    "--binary-stats-sample-size", "$BinaryStatsSampleSize"
)

if ($MaxRounds -gt 0) {
    $TrainArgs += @("--max-rounds", "$MaxRounds")
}

if ($ShowIndexProgress) {
    $TrainArgs += "--show-index-progress"
}

if ($SkipTrainedRounds) {
    $TrainArgs += "--skip-trained-rounds"
}

python @TrainArgs
exit $LASTEXITCODE