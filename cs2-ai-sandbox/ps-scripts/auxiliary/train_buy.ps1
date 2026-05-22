param (
    [string]$SplitMode = "demo",
    [switch]$ShowBuildProgress,
    [string]$SavePath = "checkpoints\buy_sklearn_v1.joblib"
)

Write-Host "Training buy model..." -ForegroundColor Cyan

$args = @(
    "cs2_ai/ml/training/train_buy_sklearn.py",
    "--split-mode", $SplitMode,
    "--save-path", $SavePath
)

if ($ShowBuildProgress) {
    $args += "--show-build-progress"
}

python @args
exit $LASTEXITCODE
