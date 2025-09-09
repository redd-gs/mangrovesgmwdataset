# Module d'Analyse des Dynamiques de Carbone des Mangroves

Ce module analyse les dynamiques de stocks de carbone dans les écosystèmes de mangroves en fonction des conditions hydrodynamiques et des cycles de marée.

## Vue d'ensemble

Les mangroves sont parmi les écosystèmes les plus efficaces pour la séquestration de carbone, mais leur capacité varie selon leur type et leur exposition aux marées :

- **Mangroves marines** : Directement exposées aux marées océaniques, taux de séquestration élevé (4-15 Mg C/ha/an)
- **Mangroves estuariennes** : Dans les zones de mélange eau douce/salée, taux le plus élevé (7-22 Mg C/ha/an)
- **Mangroves terrestres** : En bordure, moins influencées par les marées, taux plus faible (1-9 Mg C/ha/an)

## Composants Principaux

### 1. HydrodynamicModel
Modèle principal qui orchestre toute l'analyse :
- Intègre tous les sous-modules
- Exécute l'analyse complète
- Gère l'export des résultats

### 2. TimeSeriesProcessor
Traite les séries temporelles d'images satellites :
- Calcul d'indices spectraux (NDVI, NDWI, EVI)
- Détection des zones d'eau
- Calcul de la fréquence d'inondation
- Analyse des patterns temporels

### 3. TidalAnalyzer
Analyse les cycles de marée :
- Génération de données de marée théoriques
- Classification des conditions de marée
- Corrélation avec les observations satellites
- Analyse des patterns saisonniers

### 4. MangroveTypeClassifier
Classifie les mangroves selon leur exposition hydrodynamique :
- Extraction de caractéristiques hydrodynamiques
- Classification en 3 types : marine, estuarienne, terrestre
- Utilise Random Forest 
- Validation croisée spatiale

### 5. CarbonSequestrationAnalyzer
Modélise la séquestration de carbone :
- Taux de base par type de mangrove
- Ajustements environnementaux (salinité, température, marées)
- Calcul des stocks de carbone (sol + biomasse)
- Évaluation de la vulnérabilité climatique

## Utilisation

### Script Principal
```bash
cd src
python analyze_carbon_dynamics.py
```

### Utilisation Programmatique
```python
from carbon_dynamics import HydrodynamicModel

# Initialiser le modèle
model = HydrodynamicModel(
    location_lat=1.3521,    # Singapour
    location_lon=103.8198,
    pixel_size_m=10.0
)

# Exécuter l'analyse complète
results = model.run_complete_analysis(
    image_paths=image_paths,
    timestamps=timestamps,
    start_date=start_date,
    end_date=end_date,
    climate_scenarios={
        'sea_level_rise': 0.5,        # 50 cm
        'temperature_increase': 2.5,   # 2.5°C
        'storm_frequency': 1.3         # +30%
    }
)

# Exporter les résultats
model.export_results("output_directory", "analysis_prefix")
```

## Données d'Entrée

### Obligatoires
- **Images satellites** : Série temporelle d'images multispectrale (Sentinel-2)
- **Timestamps** : Dates d'acquisition des images
- **Coordonnées** : Latitude/longitude de la zone d'étude

### Optionnelles
- **Données de marée** : Si disponibles (sinon utilise un modèle théorique)
- **MNT** : Modèle numérique de terrain (DEM)
- **Données de salinité** : Mesures in-situ
- **Scénarios climatiques** : Pour l'évaluation de vulnérabilité

## Sorties

### Cartes Produites
- **Classification des mangroves** : Marine/Estuarienne/Terrestre
- **Fréquence d'inondation** : 0-1, fréquence d'inondation par pixel
- **Séquestration de carbone** : Mg C/ha/an
- **Stocks de carbone** : Mg C/ha (sol + biomasse)
- **Vulnérabilité climatique** : Index 0-1

### Rapports
- **Statistiques par type** : Surfaces, taux de séquestration, stocks
- **Analyse temporelle** : Variabilité saisonnière
- **Corrélations** : Marées vs indices de végétation
- **Vulnérabilité** : Évaluation selon scénarios climatiques

### Formats d'Export
- **JSON** : Rapports et statistiques
- **NPZ** : Cartes et arrays NumPy
- **CSV** : Données tabulaires (marées, séries temporelles)

## Méthodologie Scientifique

### Classification des Mangroves
Basée sur des caractéristiques hydrodynamiques :
- **Fréquence d'inondation** : Calculée à partir des séries temporelles
- **Amplitude de marée** : Données théoriques ou observées
- **Distance à la côte** : Proximité de l'influence marine
- **Proxy de salinité** : Indices spectraux
- **Élévation** : Hauteur au-dessus du niveau de la mer

### Modélisation de la Séquestration
Approche multi-facteurs :
1. **Taux de base** : Selon la littérature par type de mangrove
2. **Ajustements environnementaux** : Salinité, température, marées
3. **Variabilité spatiale** : Distribution statistique réaliste
4. **Variabilité temporelle** : Cycles saisonniers

### Évaluation de Vulnérabilité
Facteurs considérés :
- **Élévation du niveau de la mer** : Risque d'érosion
- **Augmentation de température** : Stress thermique
- **Fréquence des tempêtes** : Perturbations physiques

## Paramètres Scientifiques

### Taux de Séquestration (Mg C/ha/an)
- **Marine** : 8.5 ± 2.1 (plage 4.2-15.3)
- **Estuarienne** : 12.8 ± 3.5 (plage 6.8-22.1)
- **Terrestre** : 4.2 ± 1.8 (plage 1.5-8.9)

### Conditions Optimales
- **Salinité** : 15-25 ppt
- **Température** : 25-30°C
- **Amplitude de marée** : 1-3 m
- **Fréquence d'inondation** : 30-70%

## Exemples de Résultats

### Classification Typique
- Marine : 25-35% de la zone
- Estuarienne : 40-50% de la zone
- Terrestre : 15-25% de la zone

### Séquestration Moyenne
- Zone estuarienne : ~12 Mg C/ha/an
- Zone marine : ~8 Mg C/ha/an
- Zone terrestre : ~4 Mg C/ha/an

### Stocks de Carbone
- Sol : 70-80% du stock total
- Biomasse : 20-30% du stock total
- Densité : 100-300 Mg C/ha

## Validation et Incertitudes

### Sources d'Incertitudes
- **Résolution spatiale** : 10m de Sentinel-2
- **Couverture nuageuse** : Lacunes dans les séries temporelles
- **Modèle de marée** : Approximation théorique
- **Variabilité inter-annuelle** : Non prise en compte

### Validation
- Comparaison avec données in-situ disponibles
- Cohérence avec la littérature scientifique
- Validation croisée spatiale

## Applications

### Recherche Scientifique
- Quantification des services écosystémiques
- Étude des impacts du changement climatique
- Modélisation de la dynamique côtière

### Gestion Environnementale
- Planification de la conservation
- Évaluation des projets de restauration
- Surveillance environnementale

### Politiques Publiques
- Inventaires nationaux de carbone
- REDD+ (Réduction des Émissions liées à la Déforestation)
- Adaptation au changement climatique

## Dépendances

### Python
- numpy
- pandas
- scikit-learn
- scipy
- rasterio
- shapely

### Optionnelles
- matplotlib (visualisation)
- geopandas (données géospatiales)
- xarray (données multidimensionnelles)

## Références Scientifiques

1. Alongi, D.M. (2014). Carbon cycling and storage in mangrove forests. Annual Review of Marine Science, 6, 195-219.
2. Donato, D.C., et al. (2011). Mangroves among the most carbon-rich forests in the tropics. Nature Geoscience, 4(5), 293-297.
3. Twilley, R.R., et al. (2018). A comprehensive synthesis of blue carbon stocks and accumulation rates in mangroves across the Indo-Pacific. Global Change Biology, 24(10), 4454-4470.

## Contact et Support

Pour toute question ou problème :
- Vérifiez les logs d'exécution
- Consultez la documentation des modules individuels
- Adaptez les paramètres selon votre zone d'étude

## Licence

Ce module fait partie du projet d'analyse des mangroves développé pour l'étude des écosystèmes côtiers tropicaux.
