# Améliorations du Filtrage de Qualité d'Images

## Problématique

Le dataset contenait trop d'images noires ou de mauvaise qualité (avec beaucoup de couverture nuageuse), ce qui rendait l'entraînement de modèles de deep learning difficile ou impossible.

## Solutions Implémentées

### 1. Filtrage au niveau des requêtes Sentinel Hub

**Fichiers modifiés :**
- `pipeline/src/sentinel/download_s2.py`
- `pipeline/src/utils/optimized_download.py`

**Améliorations :**
- Ajout du paramètre `maxcc` (max cloud cover) dans les requêtes SentinelHubRequest
- Utilisation explicite de la valeur `MAX_CLOUD_COVER` de la configuration (10% par défaut)
- Conversion automatique du pourcentage en ratio (0-1) pour l'API

```python
# Avant
SentinelHubRequest.input_data(
    data_collection=DataCollection.SENTINEL2_L2A,
    time_interval=(start_date, end_date),
    mosaicking_order="leastCC"
)

# Après
SentinelHubRequest.input_data(
    data_collection=DataCollection.SENTINEL2_L2A,
    time_interval=(start_date, end_date),
    mosaicking_order="leastCC",
    maxcc=cfg.MAX_CLOUD_COVER / 100.0  # Filtrage explicite
)
```

### 2. Validation post-téléchargement

**Nouvelle fonction :** `validate_image_quality()`

**Critères de validation :**
- **Images entièrement noires :** Détection et rejet
- **Ratio de pixels valides :** Minimum 80% de pixels non-nuls
- **Luminosité :** Entre 0.02 et 0.95 pour éviter les images trop sombres ou saturées
- **Contraste :** Écart-type normalisé minimum de 0.1 pour éviter les images uniformes

**Paramètres configurables :**
```python
MIN_VALID_PIXELS_RATIO = 0.8     # 80% de pixels valides minimum
MIN_BRIGHTNESS_THRESHOLD = 0.02  # Éviter les images trop sombres
MAX_BRIGHTNESS_THRESHOLD = 0.95  # Éviter les images saturées
MIN_CONTRAST_RATIO = 0.1         # Contraste minimum
```

### 3. Mécanisme de retry intelligent

**Nouvelle fonction :** `download_single_with_retry()`

**Fonctionnalités :**
- **Retry automatique :** Jusqu'à 3 tentatives par défaut (configurable)
- **Fenêtres temporelles décalées :** Si une image est invalide, essaie avec d'autres dates (±30 jours)
- **Nettoyage automatique :** Suppression des fichiers invalides avant retry
- **Validation systématique :** Chaque image téléchargée est validée avant acceptation

### 4. Configuration enrichie

**Fichier modifié :** `pipeline/src/config/settings_s2.py`

**Nouveaux paramètres :**
```python
# Filtres de qualité d'image
MIN_VALID_PIXELS_RATIO = 0.8      # 80% de pixels valides minimum
MIN_BRIGHTNESS_THRESHOLD = 0.02   # Éviter les images trop sombres
MAX_BRIGHTNESS_THRESHOLD = 0.95   # Éviter les images saturées
MIN_CONTRAST_RATIO = 0.1          # Contraste minimum
RETRY_COUNT = 3                   # Nombre de tentatives si image invalide
```

## Impact sur la Qualité du Dataset

### Avant les améliorations :
- Images noires fréquentes dans les catégories "0%" (aucune mangrove)
- Images avec forte couverture nuageuse acceptées
- Pas de validation de qualité

### Après les améliorations :
- ✅ Filtrage strict de la couverture nuageuse (< 10%)
- ✅ Détection et rejet des images noires/uniformes
- ✅ Validation automatique de la luminosité et du contraste
- ✅ Retry intelligent avec fenêtres temporelles alternatives
- ✅ Nettoyage automatique des images invalides

## Utilisation

### Configuration via variables d'environnement

```bash
# Réglage de la couverture nuageuse maximum (pourcentage)
export MAX_CLOUD_COVER=5

# Réglage des seuils de qualité
export MIN_VALID_PIXELS_RATIO=0.85
export MIN_BRIGHTNESS_THRESHOLD=0.03
export RETRY_COUNT=5
```

### Utilisation programmatique

```python
from sentinel.download_s2 import download_single_with_retry

# Téléchargement avec retry et validation automatique
success = download_single_with_retry(
    bbox=my_bbox,
    output_path=Path("image.png"),
    enhanced=True,
    max_retries=3
)
```

## Monitoring et Logs

Le système fournit des logs détaillés pour suivre le processus de filtrage :

```
[INFO] Downloading band B04 with cloud cover < 10%...
[SUCCESS] Downloaded and validated band B04
[INFO] Tentative 1/3 pour image_001
[WARNING] Image invalide: Trop peu de pixels valides (65% < 80%)
[INFO] Tentative 2/3 avec fenêtre temporelle: ('2024-05-15', '2024-07-14')
[SUCCESS] Image valide: Image valide
```

## Résultats Attendus

1. **Réduction drastique des images noires** dans le dataset
2. **Amélioration de la qualité moyenne** des images
3. **Meilleure performance** des modèles de deep learning
4. **Dataset plus équilibré** entre les différentes catégories de mangroves
5. **Traçabilité complète** du processus de filtrage

Ces améliorations garantissent un dataset de haute qualité, essentiel pour l'entraînement efficace de modèles de deep learning pour la détection et classification des mangroves.