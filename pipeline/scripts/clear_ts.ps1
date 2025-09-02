[CmdletBinding()]
param(
    [string]$Dir = "pipeline/data/time_series/output",
    [switch]$Recreate,      # recrée le dossier après suppression
    [switch]$VerboseList    # affiche les fichiers supprimés
)

if (-not (Test-Path $Dir)) {
    Write-Host "[INFO] Dossier '$Dir' absent (rien à faire)."
    if ($Recreate) {
        New-Item -ItemType Directory -Path $Dir | Out-Null
        Write-Host "[INFO] Dossier recréé."
    }
    exit 0
}

$files = Get-ChildItem -File $Dir
if (-not $files) {
    Write-Host "[INFO] Aucun fichier dans $Dir."
    if ($Recreate) {
        Write-Host "[INFO] Dossier conservé."
    }
    exit 0
}

$cnt = $files.Count
if ($VerboseList) {
    $files | ForEach-Object { Write-Host " - $_" }
}
$files | Remove-Item -Force
Write-Host "[INFO] Supprimé $cnt fichiers dans $Dir"

if ($Recreate) {
    # S’assure que le dossier existe toujours
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir | Out-Null
    }
    Write-Host "[INFO] Dossier '$Dir' prêt."
}