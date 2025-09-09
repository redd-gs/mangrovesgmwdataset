try {
    # Obtenir le chemin du répertoire parent du script actuel
    $scriptRoot = $PSScriptRoot

    # Construire le chemin vers le répertoire de données
    $dataDir = Join-Path -Path $scriptRoot -ChildPath "..\data"

    # Définir les dossiers à nettoyer
    $foldersToClean = @(
        "sentinel_1\output\1_20_percent",
        "sentinel_1\output\21_40_percent"
        "sentinel_1\output\41_60_percent",
        "sentinel_1\output\61_80_percent",
        "sentinel_1\output\more_than_80_percent",
        "sentinel_1\output\no_mangroves",
        "sentinel_1\bands\1_20_percent",
        "sentinel_1\bands\21_40_percent",
        "sentinel_1\bands\41_60_percent",
        "sentinel_1\bands\61_80_percent",
        "sentinel_1\bands\more_than_80_percent",
        "sentinel_1\bands\no_mangroves"
        "sentinel_1\temporal series",
        "sentinel_2\output\1_20_percent",
        "sentinel_2\output\21_40_percent"
        "sentinel_2\output\41_60_percent",
        "sentinel_2\output\61_80_percent",
        "sentinel_2\output\more_than_80_percent",
        "sentinel_2\output\no_mangroves",
        "sentinel_2\bands\1_20_percent",
        "sentinel_2\bands\21_40_percent",
        "sentinel_2\bands\41_60_percent",
        "sentinel_2\bands\61_80_percent",
        "sentinel_2\bands\more_than_80_percent",
        "sentinel_2\bands\no_mangroves",
        "sentinel_2\temporal series"
    )
    Write-Host "Début du nettoyage du contenu des dossiers..."

    foreach ($folder in $foldersToClean) {
        $targetPath = Join-Path -Path $dataDir -ChildPath $folder
        if (Test-Path -Path $targetPath -PathType Container) {
            Write-Host "Nettoyage du contenu de : $targetPath"
            # Supprimer tous les éléments (fichiers et sous-dossiers) à l'intérieur du dossier cible
            Get-ChildItem -Path $targetPath -Recurse | Remove-Item -Recurse -Force
            Write-Host " -> Contenu nettoyé avec succès."
        }
        else {
            Write-Host "Le répertoire n'existe pas, ignoré : $targetPath"
        }
    }

    Write-Host "Nettoyage terminé."

}
catch {
    Write-Error "Une erreur est survenue : $($_.Exception.Message)"
}