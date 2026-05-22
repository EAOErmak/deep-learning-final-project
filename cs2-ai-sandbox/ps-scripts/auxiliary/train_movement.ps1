param (
    [string]$RoundsDatasetDir = "data\processed\rounds-dataset-grid",
    [int]$Epochs = 10,
    [int]$EpochsPerRound = 1,
    [int]$BatchSize = 64,
    [int]$SeqLen = 64,
    [int]$Stride = 8,
    [string]$SplitMode = "round",
    [int]$NumWorkers = -1,
    [switch]$ShowIndexProgress,
    [switch]$DisableBatchProgress,
    [int]$LogEvery = 10,
    [string]$ResumeFrom = '',
    [string]$SavePath = "checkpoints\movement_bc_v1.pt",
    [int]$MovementStatsSampleSize = 5000,
    [int]$Seed = 42,
    [int]$MaxRounds = 0,
    [switch]$SkipTrainedRounds,
    [switch]$ShuffleRounds
)

$resolvedRoundsDatasetDir = (Resolve-Path -LiteralPath $RoundsDatasetDir).Path
$datasetSubdir = Split-Path -Path $resolvedRoundsDatasetDir -Leaf
$dataDir = Split-Path -Path $resolvedRoundsDatasetDir -Parent

if ($MaxRounds -gt 0) {
    Write-Host "Movement training max rounds: $MaxRounds" -ForegroundColor Cyan
} else {
    Write-Host "Movement training max rounds: unlimited" -ForegroundColor Cyan
}
Write-Host "Movement skip trained rounds: $([bool]$SkipTrainedRounds)" -ForegroundColor Cyan
Write-Host "Movement stream dataset: $resolvedRoundsDatasetDir" -ForegroundColor DarkGray

$args = @(
    "cs2_ai/ml/training/train_movement.py",
    "--data-dir", $dataDir,
    "--dataset-subdir", $datasetSubdir,
    "--epochs", $Epochs,
    "--epochs-per-round", $EpochsPerRound,
    "--batch-size", $BatchSize,
    "--seq-len", $SeqLen,
    "--stride", $Stride,
    "--split-mode", $SplitMode,
    "--num-workers", $NumWorkers,
    "--log-every", $LogEvery,
    "--movement-feature-mode", "solo_grid",
    "--movement-stats-sample-size", $MovementStatsSampleSize,
    "--save-path", $SavePath,
    "--stream-by-round",
    "--seed", $Seed
)

if ($ShowIndexProgress) {
    $args += "--show-index-progress"
}

if ($DisableBatchProgress) {
    $args += "--disable-batch-progress"
}

if ($ShuffleRounds) {
    $args += "--shuffle-rounds"
}

if ($SkipTrainedRounds) {
    $args += "--skip-trained-rounds"
}

if ($MaxRounds -gt 0) {
    $args += @("--max-rounds", "$MaxRounds")
}

if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $args += @("--resume-from", $ResumeFrom)
}

python @args
exit $LASTEXITCODE
