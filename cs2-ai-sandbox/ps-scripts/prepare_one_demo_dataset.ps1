param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [switch]$RequireBuyTicks,
    [switch]$RequireGridRounds,
    [string]$TargetRoot = 'dataset-test',
    [string]$ProcessedRoot = 'data\processed'
)

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

$projectRoot = $PSScriptRoot
$demoBase = Resolve-DemoBaseName -Name $DemoName
$datasetDir = Join-Path $projectRoot $TargetRoot
$targetDir = Join-Path $datasetDir $demoBase
$processedDir = Join-Path $projectRoot $ProcessedRoot

$targetRoundsDir = Join-Path $targetDir 'rounds-dataset'
$targetGridRoundsDir = Join-Path $targetDir 'rounds-dataset-grid'
$targetBuyDir = Join-Path $targetDir 'clean_buy_ticks'

New-Item -ItemType Directory -Force -Path $targetRoundsDir | Out-Null
New-Item -ItemType Directory -Force -Path $targetGridRoundsDir | Out-Null
New-Item -ItemType Directory -Force -Path $targetBuyDir | Out-Null

$sourceRoundsDemoDir = Join-Path $processedDir ("rounds-dataset\{0}" -f $demoBase)
if (-not (Test-Path -LiteralPath $sourceRoundsDemoDir)) {
    throw "Rounds dataset directory not found for demo '$demoBase'. Expected: $sourceRoundsDemoDir"
}
Copy-Item -LiteralPath $sourceRoundsDemoDir -Destination $targetRoundsDir -Recurse -Force

$sourceRoundsManifest = Join-Path $processedDir 'rounds-dataset\manifest.json'
$sourceRoundsSummary = Join-Path $processedDir 'rounds-dataset\rounds_summary.csv'
if (Test-Path -LiteralPath $sourceRoundsManifest) {
    Copy-Item -LiteralPath $sourceRoundsManifest -Destination (Join-Path $targetRoundsDir 'manifest.json') -Force
}
if (Test-Path -LiteralPath $sourceRoundsSummary) {
    Copy-Item -LiteralPath $sourceRoundsSummary -Destination (Join-Path $targetRoundsDir 'rounds_summary.csv') -Force
}

$sourceGridDemoDir = Join-Path $processedDir ("rounds-dataset-grid\{0}" -f $demoBase)
if (Test-Path -LiteralPath $sourceGridDemoDir) {
    Copy-Item -LiteralPath $sourceGridDemoDir -Destination $targetGridRoundsDir -Recurse -Force
    $sourceGridManifest = Join-Path $processedDir 'rounds-dataset-grid\manifest.json'
    $sourceGridSummary = Join-Path $processedDir 'rounds-dataset-grid\rounds_summary.csv'
    if (Test-Path -LiteralPath $sourceGridManifest) {
        Copy-Item -LiteralPath $sourceGridManifest -Destination (Join-Path $targetGridRoundsDir 'manifest.json') -Force
    }
    if (Test-Path -LiteralPath $sourceGridSummary) {
        Copy-Item -LiteralPath $sourceGridSummary -Destination (Join-Path $targetGridRoundsDir 'rounds_summary.csv') -Force
    }
} elseif ($RequireGridRounds) {
    throw "Grid-labelled rounds dataset not found for demo '$demoBase'. Expected: $sourceGridDemoDir"
}

$sourceBuy = Join-Path $projectRoot ("dataset\clean_buy_ticks\{0}_buy_ticks.parquet" -f $demoBase)
if (Test-Path -LiteralPath $sourceBuy) {
    Copy-Item -LiteralPath $sourceBuy -Destination $targetBuyDir -Force
} elseif ($RequireBuyTicks) {
    throw "Buy parquet not found for demo '$demoBase'. Expected: $sourceBuy"
}

Write-Host "Prepared one-demo dataset: $targetDir" -ForegroundColor DarkCyan
Write-Output $targetDir
