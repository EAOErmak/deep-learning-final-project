param (
    [string]$RoundsDatasetDir = "data\processed\rounds-dataset",
    [string]$OutputDir = "data\processed\rounds-dataset-grid",
    [string]$Map = "de_dust2",
    [int]$LookaheadTicks = 10,
    [int]$MinTargetDistance = 75,
    [int]$Workers = 1
)

$ErrorActionPreference = 'Stop'

Write-Host "Labelling rounds dataset with grid targets..." -ForegroundColor Cyan
Write-Host "RoundsDatasetDir=$RoundsDatasetDir OutputDir=$OutputDir Map=$Map LookaheadTicks=$LookaheadTicks MinTargetDistance=$MinTargetDistance Workers=$Workers" -ForegroundColor DarkGray

$args = @(
    "-m", "cs2_ai.preprocessing.label_rounds_grid",
    "--rounds-dataset-dir", $RoundsDatasetDir,
    "--output-dir", $OutputDir,
    "--map", $Map,
    "--lookahead-ticks", $LookaheadTicks,
    "--min-target-distance", $MinTargetDistance,
    "--workers", $Workers
)

python @args
exit $LASTEXITCODE
