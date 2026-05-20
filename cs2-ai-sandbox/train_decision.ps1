param (
    [int]$Epochs = 3,
    [int]$BatchSize = 32,
    [string]$SplitMode = "demo",
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 0,
    [string]$SavePath = "checkpoints\decision_dqn.pt",
    [switch]$DisableTensorboard
)

Write-Host "Training offline Decision DQN model..." -ForegroundColor Cyan

$args = @(
    "cs2_ai/ml/training/train_decision_offline.py",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--split-mode", $SplitMode,
    "--save-path", $SavePath
)

if ($MaxSamples -gt 0) {
    $args += "--max-samples"
    $args += $MaxSamples
}
if ($MaxSamplesPerDemo -gt 0) {
    $args += "--max-samples-per-demo"
    $args += $MaxSamplesPerDemo
}
if ($DisableTensorboard) {
    $args += "--disable-tensorboard"
}

python @args
exit $LASTEXITCODE
