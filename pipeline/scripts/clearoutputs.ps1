try {
    # Obtenir le chemin du répertoire parent du script actuel
    $scriptRoot = $PSScriptRoot

    # Construire le chemin vers le répertoire de données
    $dataDir = Join-Path -Path $scriptRoot -ChildPath "..\data"

    # Définir les dossiers à nettoyer
    $foldersToClean = @(
        "sentinel_1\output\1-20%",
        "sentinel_1\output\20-40%",
        "sentinel_1\output\410-60%",
        "sentinel_1\output\60-80%",
        "sentinel_1\output\80-100%",
        "sentinel_1\output\0%",
        "sentinel_1\bands\1-20%",
        "sentinel_1\bands\20-40%",
        "sentinel_1\bands\40-60%",
        "sentinel_1\bands\60-80%",
        "sentinel_1\bands\80-100%",
        "sentinel_1\bands\0%",
        "sentinel_1\temporal series",
        "sentinel_2\output\1-20%",
        "sentinel_2\output\20-40%",
        "sentinel_2\output\40-60%",
        "sentinel_2\output\60-80%",
        "sentinel_2\output\80-100%",
        "sentinel_2\output\0%",
        "sentinel_2\bands\1-20%",
        "sentinel_2\bands\20-40%",
        "sentinel_2\bands\40-60%",
        "sentinel_2\bands\60-80%",
        "sentinel_2\bands\80-100%",
        "sentinel_2\bands\0%",
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