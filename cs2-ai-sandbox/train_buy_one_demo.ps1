param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [switch]$ShowBuildProgress,
    [string]$SavePath = ''
)

Write-Host "Training buy model on one demo..." -ForegroundColor Cyan
$datasetDir = & (Join-Path $PSScriptRoot 'prepare_one_demo_dataset.ps1') -DemoName $DemoName -RequireBuyTicks
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$demoBase = Split-Path $datasetDir -Leaf
if ([string]::IsNullOrWhiteSpace($SavePath)) {
    $SavePath = "checkpoints\buy_$demoBase.joblib"
}

$args = @(
    "cs2_ai/ml/training/train_buy_sklearn.py",
    "--dataset-dir", $datasetDir,
    "--split-mode", "round",
    "--save-path", $SavePath
)
if ($ShowBuildProgress) { $args += "--show-build-progress" }

python @args
exit $LASTEXITCODE
