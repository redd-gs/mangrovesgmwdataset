"""
Processeur de séries temporelles pour l'analyse des mangroves.

Ce module traite les séries temporelles d'images satellites pour extraire
les variations temporelles des caractéristiques des mangroves.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import rasterio
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """
    Classe pour traiter les séries temporelles d'images de mangroves.
    """
    
    def __init__(self, temporal_resolution_days: int = 10):
        """
        Initialise le processeur de séries temporelles.
        
        Args:
            temporal_resolution_days: Résolution temporelle en jours pour l'analyse
        """
        self.temporal_resolution = temporal_resolution_days
        self.scaler = StandardScaler()
        
    def load_time_series(self, image_paths: List[str], timestamps: List[datetime]) -> Dict:
        """
        Charge une série temporelle d'images.
        
        Args:
            image_paths: Liste des chemins vers les images
            timestamps: Liste des timestamps correspondants
            
        Returns:
            Dictionnaire contenant les données temporelles
        """
        time_series_data = {
            'timestamps': timestamps,
            'images': [],
            'metadata': []
        }
        
        for i, (path, timestamp) in enumerate(zip(image_paths, timestamps)):
            try:
                # Gestion spéciale pour les données synthétiques
                if path == "synthetic_data":
                    # Créer des données synthétiques
                    height, width = 100, 100
                    # Simuler 4 bandes spectrales (B02, B03, B04, B08)
                    image_data = np.random.rand(4, height, width).astype(np.float32)
                    # Ajuster les valeurs pour être plus réalistes (réflectance 0-1)
                    image_data = image_data * 0.3 + 0.05  # Valeurs entre 0.05 et 0.35
                    
                    metadata = {
                        'path': path,
                        'timestamp': timestamp,
                        'shape': image_data.shape,
                        'crs': 'EPSG:4326',  # WGS84
                        'transform': None,
                        'synthetic': True
                    }
                else:
                    # Chargement normal avec rasterio
                    with rasterio.open(path) as src:
                        image_data = src.read()
                        metadata = {
                            'path': path,
                            'timestamp': timestamp,
                            'shape': image_data.shape,
                            'crs': src.crs,
                            'transform': src.transform,
                            'synthetic': False
                        }
                    
                time_series_data['images'].append(image_data)
                time_series_data['metadata'].append(metadata)
                    
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {path}: {e}")
                continue
                
        return time_series_data
    
    def extract_spectral_indices(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extrait les indices spectraux utiles pour l'analyse des mangroves.
        
        Args:
            image_data: Données d'image (B, H, W) où B = bandes spectrales
            
        Returns:
            Dictionnaire des indices spectraux calculés
        """
        # Supposons que les bandes sont dans l'ordre: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
        if image_data.shape[0] < 3:
            raise ValueError("Au moins 3 bandes spectrales sont nécessaires")
            
        blue = image_data[0].astype(np.float32)
        green = image_data[1].astype(np.float32) 
        red = image_data[2].astype(np.float32)
        
        # Si NIR disponible
        nir = image_data[3].astype(np.float32) if image_data.shape[0] > 3 else None
        
        indices = {}
        
        # NDVI (si NIR disponible)
        if nir is not None:
            indices['ndvi'] = np.divide(
                nir - red,
                nir + red,
                out=np.zeros_like(nir),
                where=(nir + red) != 0
            )
        
        # NDWI (Normalized Difference Water Index)
        if nir is not None:
            indices['ndwi'] = np.divide(
                green - nir,
                green + nir,
                out=np.zeros_like(green),
                where=(green + nir) != 0
            )
        
        # EVI (Enhanced Vegetation Index)
        if nir is not None:
            indices['evi'] = np.divide(
                2.5 * (nir - red),
                nir + 6 * red - 7.5 * blue + 1,
                out=np.zeros_like(nir),
                where=(nir + 6 * red - 7.5 * blue + 1) != 0
            )
        
        # Green-Red Vegetation Index
        indices['grvi'] = np.divide(
            green - red,
            green + red,
            out=np.zeros_like(green),
            where=(green + red) != 0
        )
        
        # Blue-Green ratio (indicateur de turbidité de l'eau)
        indices['bg_ratio'] = np.divide(
            blue,
            green,
            out=np.ones_like(blue),
            where=green != 0
        )
        
        return indices
    
    def detect_water_areas(self, spectral_indices: Dict[str, np.ndarray], 
                          threshold_ndwi: float = 0.3) -> np.ndarray:
        """
        Détecte les zones d'eau dans l'image.
        
        Args:
            spectral_indices: Indices spectraux calculés
            threshold_ndwi: Seuil NDWI pour la détection d'eau
            
        Returns:
            Masque binaire des zones d'eau
        """
        if 'ndwi' in spectral_indices:
            water_mask = spectral_indices['ndwi'] > threshold_ndwi
        else:
            # Fallback avec d'autres indices
            water_mask = spectral_indices['bg_ratio'] > 1.2
            
        # Filtrage morphologique pour nettoyer le masque
        water_mask = ndimage.binary_opening(water_mask, structure=np.ones((3, 3)))
        water_mask = ndimage.binary_closing(water_mask, structure=np.ones((5, 5)))
        
        return water_mask
    
    def calculate_inundation_frequency(self, time_series_data: Dict) -> np.ndarray:
        """
        Calcule la fréquence d'inondation pour chaque pixel.
        
        Args:
            time_series_data: Données de série temporelle
            
        Returns:
            Carte de fréquence d'inondation (0-1)
        """
        water_masks = []
        
        for image_data in time_series_data['images']:
            indices = self.extract_spectral_indices(image_data)
            water_mask = self.detect_water_areas(indices)
            water_masks.append(water_mask)
        
        if not water_masks:
            return np.zeros((100, 100))  # Default empty array
            
        # Calculer la fréquence d'inondation
        water_stack = np.stack(water_masks, axis=0)
        inundation_frequency = np.mean(water_stack, axis=0)
        
        return inundation_frequency
    
    def analyze_temporal_patterns(self, time_series_data: Dict) -> Dict:
        """
        Analyse les patterns temporels dans la série.
        
        Args:
            time_series_data: Données de série temporelle
            
        Returns:
            Dictionnaire des métriques temporelles
        """
        timestamps = time_series_data['timestamps']
        
        # Extraire les indices pour chaque image
        temporal_indices = {
            'ndvi': [],
            'ndwi': [],
            'evi': [],
            'timestamps': timestamps
        }
        
        for image_data in time_series_data['images']:
            indices = self.extract_spectral_indices(image_data)
            
            # Moyennes spatiales des indices
            for key in ['ndvi', 'ndwi', 'evi']:
                if key in indices:
                    mean_value = np.nanmean(indices[key])
                    temporal_indices[key].append(mean_value)
                else:
                    temporal_indices[key].append(np.nan)
        
        # Créer un DataFrame pour l'analyse
        df = pd.DataFrame(temporal_indices)
        df['timestamp'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamp')
        
        # Calculer les métriques temporelles
        metrics = {
            'temporal_variability': {},
            'seasonal_patterns': {},
            'trend_analysis': {}
        }
        
        for index_name in ['ndvi', 'ndwi', 'evi']:
            if not df[index_name].isna().all():
                metrics['temporal_variability'][index_name] = {
                    'std': df[index_name].std(),
                    'cv': df[index_name].std() / df[index_name].mean() if df[index_name].mean() != 0 else 0,
                    'range': df[index_name].max() - df[index_name].min()
                }
        
        return metrics
    
    def interpolate_missing_data(self, time_series_data: Dict, 
                               method: str = 'linear') -> Dict:
        """
        Interpole les données manquantes dans la série temporelle.
        
        Args:
            time_series_data: Données de série temporelle
            method: Méthode d'interpolation ('linear', 'cubic', 'nearest')
            
        Returns:
            Données interpolées
        """
        # Cette fonction pourrait être développée pour interpoler
        # les images manquantes dans la série temporelle
        logger.info(f"Interpolation des données manquantes avec la méthode: {method}")
        return time_series_data
    
    def export_time_series_metrics(self, metrics: Dict, output_path: str):
        """
        Exporte les métriques de série temporelle vers un fichier.
        
        Args:
            metrics: Métriques calculées
            output_path: Chemin de sortie
        """
        import json
        
        # Convertir les arrays numpy en listes pour la sérialisation JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        serializable_metrics = convert_numpy(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        logger.info(f"Métriques exportées vers: {output_path}")
