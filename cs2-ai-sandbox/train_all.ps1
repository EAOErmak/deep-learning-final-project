param (
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [string]$SplitMode = "demo",
    [int]$NumWorkers = -1,
    [int]$MaxSamples = 0,
    [int]$MaxSamplesPerDemo = 0,
    [switch]$ShowIndexProgress,
    [switch]$ShowBuildProgress
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Full Neural AI Pipeline Training" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

Write-Host "`n[1/5] Training Enemy Tracker..." -ForegroundColor Yellow
$trackerArgs = @{
    Epochs = $Epochs
    BatchSize = $BatchSize
    SplitMode = $SplitMode
    NumWorkers = $NumWorkers
}
if ($MaxSamples -gt 0) {
    $trackerArgs.Add("MaxSamples", $MaxSamples)
}
if ($MaxSamplesPerDemo -gt 0) {
    $trackerArgs.Add("MaxSamplesPerDemo", $MaxSamplesPerDemo)
}
if ($ShowIndexProgress) {
    $trackerArgs.Add("ShowIndexProgress", $true)
}
& "./train_enemy_tracker.ps1" @trackerArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Enemy Tracker training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[2/5] Training Movement Model..." -ForegroundColor Yellow
$movementArgs = @{
    Epochs = $Epochs
    BatchSize = $BatchSize
    SplitMode = $SplitMode
    NumWorkers = $NumWorkers
}
if ($MaxSamples -gt 0) {
    $movementArgs.Add("MaxSamples", $MaxSamples)
}
if ($MaxSamplesPerDemo -gt 0) {
    $movementArgs.Add("MaxSamplesPerDemo", $MaxSamplesPerDemo)
}
if ($ShowIndexProgress) {
    $movementArgs.Add("ShowIndexProgress", $true)
}
& "./train_movement.ps1" @movementArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Movement training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[3/5] Training Aim Model..." -ForegroundColor Yellow
$aimArgs = @{
    Epochs = $Epochs
    BatchSize = $BatchSize
    SplitMode = $SplitMode
    NumWorkers = $NumWorkers
}
if ($MaxSamples -gt 0) {
    $aimArgs.Add("MaxSamples", $MaxSamples)
}
if ($MaxSamplesPerDemo -gt 0) {
    $aimArgs.Add("MaxSamplesPerDemo", $MaxSamplesPerDemo)
}
if ($ShowIndexProgress) {
    $aimArgs.Add("ShowIndexProgress", $true)
}
& "./train_aim.ps1" @aimArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Aim training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[4/5] Training Buy Model..." -ForegroundColor Yellow
$buyArgs = @{
    SplitMode = $SplitMode
}
if ($ShowBuildProgress) {
    $buyArgs.Add("ShowBuildProgress", $true)
}
& "./train_buy.ps1" @buyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Buy training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[5/5] Training Decision DQN Model..." -ForegroundColor Yellow
$decisionArgs = @{
    Epochs = $Epochs
    BatchSize = $BatchSize
    SplitMode = $SplitMode
}
if ($MaxSamples -gt 0) {
    $decisionArgs.Add("MaxSamples", $MaxSamples)
}
if ($MaxSamplesPerDemo -gt 0) {
    $decisionArgs.Add("MaxSamplesPerDemo", $MaxSamplesPerDemo)
}
& "./train_decision.ps1" @decisionArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Decision DQN training failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "All models trained successfully!" -ForegroundColor Green
Write-Host "You can view the metrics in Tensorboard by running:" -ForegroundColor Green
Write-Host "tensorboard --logdir=runs" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Green
