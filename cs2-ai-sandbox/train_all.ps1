param (
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = 0,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 0,
    [switch]$ShowIndexProgress,
    [switch]$ShowBuildProgress
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Full Neural AI Pipeline Training" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Training Enemy Tracker..." -ForegroundColor Yellow
$trackerArgs = @(
    "./train_enemy_tracker.ps1",
    "-Epochs", $Epochs,
    "-BatchSize", $BatchSize,
    "-SplitMode", $SplitMode,
    "-NumWorkers", $NumWorkers
)
if ($MaxSamples -gt 0) {
    $trackerArgs += "-MaxSamples"
    $trackerArgs += $MaxSamples
}
if ($MaxSamplesPerDemo -gt 0) {
    $trackerArgs += "-MaxSamplesPerDemo"
    $trackerArgs += $MaxSamplesPerDemo
}
if ($ShowIndexProgress) {
    $trackerArgs += "-ShowIndexProgress"
}
& $trackerArgs[0] @trackerArgs[1..($trackerArgs.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    Write-Host "Enemy Tracker training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[2/4] Training Movement Model..." -ForegroundColor Yellow
$movementArgs = @(
    "./train_movement.ps1",
    "-Epochs", $Epochs,
    "-BatchSize", $BatchSize,
    "-SplitMode", $SplitMode,
    "-NumWorkers", $NumWorkers
)
if ($MaxSamples -gt 0) {
    $movementArgs += "-MaxSamples"
    $movementArgs += $MaxSamples
}
if ($MaxSamplesPerDemo -gt 0) {
    $movementArgs += "-MaxSamplesPerDemo"
    $movementArgs += $MaxSamplesPerDemo
}
if ($ShowIndexProgress) {
    $movementArgs += "-ShowIndexProgress"
}
& $movementArgs[0] @movementArgs[1..($movementArgs.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    Write-Host "Movement training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[3/4] Training Aim Model..." -ForegroundColor Yellow
$aimArgs = @(
    "./train_aim.ps1",
    "-Epochs", $Epochs,
    "-BatchSize", $BatchSize,
    "-SplitMode", $SplitMode,
    "-NumWorkers", $NumWorkers
)
if ($MaxSamples -gt 0) {
    $aimArgs += "-MaxSamples"
    $aimArgs += $MaxSamples
}
if ($MaxSamplesPerDemo -gt 0) {
    $aimArgs += "-MaxSamplesPerDemo"
    $aimArgs += $MaxSamplesPerDemo
}
if ($ShowIndexProgress) {
    $aimArgs += "-ShowIndexProgress"
}
& $aimArgs[0] @aimArgs[1..($aimArgs.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    Write-Host "Aim training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[4/4] Training Buy Model..." -ForegroundColor Yellow
$buyArgs = @(
    "./train_buy.ps1",
    "-SplitMode", $SplitMode
)
if ($ShowBuildProgress) {
    $buyArgs += "-ShowBuildProgress"
}
& $buyArgs[0] @buyArgs[1..($buyArgs.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    Write-Host "Buy training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "All models trained successfully!" -ForegroundColor Green
Write-Host "You can view the metrics in Tensorboard by running:" -ForegroundColor Green
Write-Host "tensorboard --logdir=runs" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Green
