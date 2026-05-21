param (
    [Parameter(Mandatory = $true)]
    [string]$DemoName,
    [switch]$RequireBuyTicks,
    [string]$TargetRoot = 'dataset-test'
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
$targetPlayDir = Join-Path $targetDir 'clean_play_ticks'
$targetBuyDir = Join-Path $targetDir 'clean_buy_ticks'

New-Item -ItemType Directory -Force -Path $targetPlayDir | Out-Null
New-Item -ItemType Directory -Force -Path $targetBuyDir | Out-Null

$sourcePlay = Join-Path $projectRoot ("dataset\\clean_play_ticks\\{0}_play_ticks.parquet" -f $demoBase)
if (-not (Test-Path -LiteralPath $sourcePlay)) {
    throw "Play parquet not found for demo '$demoBase'. Expected: $sourcePlay"
}
Copy-Item -LiteralPath $sourcePlay -Destination $targetPlayDir -Force

$sourceBuy = Join-Path $projectRoot ("dataset\\clean_buy_ticks\\{0}_buy_ticks.parquet" -f $demoBase)
if (Test-Path -LiteralPath $sourceBuy) {
    Copy-Item -LiteralPath $sourceBuy -Destination $targetBuyDir -Force
} elseif ($RequireBuyTicks) {
    throw "Buy parquet not found for demo '$demoBase'. Expected: $sourceBuy"
}

Write-Host "Prepared one-demo dataset: $targetDir" -ForegroundColor DarkCyan
Write-Output $targetDir


