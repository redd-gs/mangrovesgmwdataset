[CmdletBinding()]
param(
    [string]$Dir1 = "$PSScriptRoot/../data/sentinel_1/output",
    [string]$Dir2 = "$PSScriptRoot/../data/sentinel_1/temp",
    [string]$Dir3 = "$PSScriptRoot/../data/sentinel_1/temporal_series"
)

if (-not (Test-Path $Dir1)) {
    Write-Host "[INFO] Dossier '$Dir1' absent (rien à faire)."
    exit 0
}
if (-not (Test-Path $Dir2)) {
    Write-Host "[INFO] Dossier '$Dir2' absent (rien à faire)."
    exit 0
}
if (-not (Test-Path $Dir3)) {
    Write-Host "[INFO] Dossier '$Dir3' absent (rien à faire)."
    exit 0
}

$items1 = Get-ChildItem $Dir1
if (-not $items1) {
    Write-Host "[INFO] Aucun élément dans $Dir1."
}

$items2 = Get-ChildItem $Dir2
if (-not $items2) {
    Write-Host "[INFO] Aucun élément dans $Dir2."
}

$items3 = Get-ChildItem $Dir3
if (-not $items3) {
    Write-Host "[INFO] Aucun élément dans $Dir3."
}

$cnt = $items1.Count
if ($VerboseList) {
    $items1 | ForEach-Object { Write-Host " - $_" }
}
$items1 | Remove-Item -Force -Recurse
Write-Host "[INFO] Supprimé $cnt éléments dans $Dir1."


$cnt = $items2.Count
if ($VerboseList) {
    $items2 | ForEach-Object { Write-Host " - $_" }
}
$items2 | Remove-Item -Force -Recurse
Write-Host "[INFO] Supprimé $cnt éléments dans $Dir2."


$cnt = $items3.Count
if ($VerboseList) {
    $items3 | ForEach-Object { Write-Host " - $_" }
}
$items3 | Remove-Item -Force -Recurse
Write-Host "[INFO] Supprimé $cnt éléments dans $Dir3."
