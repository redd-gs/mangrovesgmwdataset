"""
Script pour créer tous les dossiers nécessaires pour les catégories de mangroves.
"""

import os
from pathlib import Path

# Catégories de couverture de mangroves
COVERAGE_CATEGORIES = {
    "no_mangroves": (0, 0),
    "1_20_percent": (1, 20),
    "21_40_percent": (21, 40),
    "41_60_percent": (41, 60),
    "61_80_percent": (61, 80),
    "more_than_80_percent": (81, 100),
}

def create_category_directories():
    """
    Crée tous les dossiers nécessaires pour chaque catégorie de mangroves.
    """
    print("=== Création des dossiers pour les catégories de mangroves ===\n")
    
    # Dossiers de base
    base_dir = Path(__file__).parent.parent / "data" / "sentinel_2"
    bands_dir = base_dir / "bands"
    output_dir = base_dir / "output"
    temporal_series_dir = base_dir / "temporal_series"
    
    directories_created = []
    
    for category_name in COVERAGE_CATEGORIES.keys():
        print(f"Création des dossiers pour la catégorie: {category_name}")
        
        # Créer les dossiers pour chaque catégorie
        category_dirs = [
            bands_dir / category_name,
            output_dir / category_name,
            temporal_series_dir / category_name,
        ]
        
        for dir_path in category_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ {dir_path}")
                directories_created.append(str(dir_path))
            except Exception as e:
                print(f"  ✗ Erreur lors de la création de {dir_path}: {e}")
    
    print(f"\n=== Résumé ===")
    print(f"Dossiers créés: {len(directories_created)}")
    print(f"Catégories configurées: {len(COVERAGE_CATEGORIES)}")
    
    # Vérifier la structure
    print(f"\n=== Structure créée ===")
    for category in COVERAGE_CATEGORIES.keys():
        print(f"\n{category}:")
        print(f"  - Bands: {bands_dir / category}")
        print(f"  - Output: {output_dir / category}")
        print(f"  - Temporal series: {temporal_series_dir / category}")
    
    return directories_created

def verify_directory_structure():
    """
    Vérifie que tous les dossiers nécessaires existent.
    """
    print("\n=== Vérification de la structure des dossiers ===")
    
    base_dir = Path(__file__).parent.parent / "data" / "sentinel_2"
    bands_dir = base_dir / "bands"
    output_dir = base_dir / "output"
    temporal_series_dir = base_dir / "temporal_series"
    
    all_good = True
    
    for category_name in COVERAGE_CATEGORIES.keys():
        category_dirs = [
            bands_dir / category_name,
            output_dir / category_name,
            temporal_series_dir / category_name,
        ]
        
        for dir_path in category_dirs:
            if dir_path.exists():
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ MANQUANT: {dir_path}")
                all_good = False
    
    if all_good:
        print("\n✓ Tous les dossiers nécessaires sont présents!")
    else:
        print("\n✗ Certains dossiers sont manquants.")
    
    return all_good

def show_directory_tree():
    """
    Affiche l'arbre des dossiers créés.
    """
    print("\n=== Arbre des dossiers sentinel_2 ===")
    
    base_dir = Path(__file__).parent.parent / "data" / "sentinel_2"
    
    if not base_dir.exists():
        print(f"Le dossier {base_dir} n'existe pas!")
        return
    
    def print_tree(path, prefix=""):
        """Fonction récursive pour afficher l'arbre des dossiers."""
        if path.is_dir():
            print(f"{prefix}📁 {path.name}/")
            try:
                # Lister les sous-dossiers
                subdirs = [p for p in path.iterdir() if p.is_dir()]
                subdirs.sort()
                
                for i, subdir in enumerate(subdirs):
                    is_last = i == len(subdirs) - 1
                    extension = "└── " if is_last else "├── "
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print(f"{prefix}{extension}📁 {subdir.name}/")
                    
            except PermissionError:
                print(f"{prefix}    [Permission refusée]")
    
    print_tree(base_dir)

if __name__ == "__main__":
    print("Script de création des dossiers pour le dataset de mangroves\n")
    
    # Créer les dossiers
    created_dirs = create_category_directories()
    
    # Vérifier la structure
    verify_directory_structure()
    
    # Afficher l'arbre
    show_directory_tree()
    
    print(f"\n{'='*60}")
    print("✓ Création des dossiers terminée avec succès!")
    print("✓ Vous pouvez maintenant lancer le script de génération du dataset.")
    print("="*60)
