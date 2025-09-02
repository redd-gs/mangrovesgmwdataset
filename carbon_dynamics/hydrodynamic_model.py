"""
Modèle hydrodynamique des mangroves intégrant les marées et la séquestration de carbone.

Ce module combine tous les composants pour créer un modèle complet d'analyse
des dynamiques de carbone en fonction des conditions hydrodynamiques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import json
import os

from .time_series_processor import TimeSeriesProcessor
from .tidal_analysis import TidalAnalyzer
from .mangrove_classifier import MangroveTypeClassifier
from .carbon_sequestration import CarbonSequestrationAnalyzer

logger = logging.getLogger(__name__)


class HydrodynamicModel:
    """
    Modèle hydrodynamique intégré pour l'analyse des mangroves et du carbone.
    """
    
    def __init__(self, location_lat: float, location_lon: float, 
                 pixel_size_m: float = 10.0):
        """
        Initialise le modèle hydrodynamique.
        
        Args:
            location_lat: Latitude de la zone d'étude
            location_lon: Longitude de la zone d'étude
            pixel_size_m: Taille du pixel en mètres
        """
        self.location_lat = location_lat
        self.location_lon = location_lon
        self.pixel_size_m = pixel_size_m
        self.area_per_pixel_ha = (pixel_size_m * pixel_size_m) / 10000  # Conversion en hectares
        
        # Initialiser les modules
        self.time_series_processor = TimeSeriesProcessor()
        self.tidal_analyzer = TidalAnalyzer(location_lat, location_lon)
        self.mangrove_classifier = MangroveTypeClassifier()
        self.carbon_analyzer = CarbonSequestrationAnalyzer()
        
        # Stockage des résultats
        self.results = {}
        
    def load_and_process_data(self, 
                            image_paths: List[str],
                            timestamps: List[datetime],
                            start_analysis: datetime,
                            end_analysis: datetime) -> Dict:
        """
        Charge et traite les données de série temporelle et de marées.
        
        Args:
            image_paths: Chemins vers les images satellites
            timestamps: Timestamps des images
            start_analysis: Date de début d'analyse
            end_analysis: Date de fin d'analyse
            
        Returns:
            Dictionnaire des données traitées
        """
        logger.info("Chargement et traitement des données...")
        
        # Traiter les séries temporelles d'images
        time_series_data = self.time_series_processor.load_time_series(
            image_paths, timestamps
        )
        
        # Générer les données de marée
        tidal_data = self.tidal_analyzer.get_tidal_data_theoretical(
            start_analysis, end_analysis
        )
        
        # Classifier les conditions de marée
        tidal_classified = self.tidal_analyzer.classify_tidal_conditions(tidal_data)
        tidal_with_range = self.tidal_analyzer.calculate_tidal_range(tidal_classified)
        
        # Calculer la fréquence d'inondation
        inundation_frequency = self.time_series_processor.calculate_inundation_frequency(
            time_series_data
        )
        
        # Analyser les patterns temporels
        temporal_patterns = self.time_series_processor.analyze_temporal_patterns(
            time_series_data
        )
        
        processed_data = {
            'time_series_data': time_series_data,
            'tidal_data': tidal_with_range,
            'inundation_frequency': inundation_frequency,
            'temporal_patterns': temporal_patterns
        }
        
        self.results['processed_data'] = processed_data
        logger.info("Traitement des données terminé")
        
        return processed_data
    
    def create_hydrodynamic_maps(self, processed_data: Dict) -> Dict:
        """
        Crée les cartes hydrodynamiques nécessaires à l'analyse.
        
        Args:
            processed_data: Données traitées
            
        Returns:
            Dictionnaire des cartes hydrodynamiques
        """
        logger.info("Création des cartes hydrodynamiques...")
        
        inundation_freq = processed_data['inundation_frequency']
        tidal_data = processed_data['tidal_data']
        
        # Créer une carte d'amplitude de marée constante (simplifié)
        tidal_range_map = np.full_like(
            inundation_freq, 
            tidal_data['tidal_range'].mean(), 
            dtype=np.float32
        )
        
        # Créer une carte de distance à la côte (simplifié - gradient depuis le bord)
        distance_to_coast = self._create_distance_to_coast_map(inundation_freq.shape)
        
        # Créer un proxy de salinité basé sur l'inondation et la distance
        salinity_proxy = self._create_salinity_proxy(
            inundation_freq, distance_to_coast
        )
        
        # Créer une carte d'élévation simplifiée
        elevation_map = self._create_elevation_map(
            inundation_freq.shape, distance_to_coast
        )
        
        hydrodynamic_maps = {
            'inundation_frequency': inundation_freq,
            'tidal_range': tidal_range_map,
            'distance_to_coast': distance_to_coast,
            'salinity_proxy': salinity_proxy,
            'elevation': elevation_map
        }
        
        self.results['hydrodynamic_maps'] = hydrodynamic_maps
        logger.info("Cartes hydrodynamiques créées")
        
        return hydrodynamic_maps
    
    def _create_distance_to_coast_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Crée une carte simplifiée de distance à la côte.
        
        Args:
            shape: Forme de la carte (height, width)
            
        Returns:
            Carte de distance (mètres)
        """
        height, width = shape
        
        # Créer un gradient de distance depuis le bord inférieur (supposé côte)
        y_coords = np.arange(height)
        distance_map = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                # Distance au bord le plus proche (côte supposée en bas)
                dist_to_bottom = (height - 1 - i) * self.pixel_size_m
                dist_to_sides = min(j, width - 1 - j) * self.pixel_size_m
                
                # Distance euclidienne approximative
                distance_map[i, j] = np.sqrt(dist_to_bottom**2 + dist_to_sides**2)
        
        return distance_map.astype(np.float32)
    
    def _create_salinity_proxy(self, inundation_freq: np.ndarray, 
                             distance_coast: np.ndarray) -> np.ndarray:
        """
        Crée un proxy de salinité basé sur l'inondation et la distance à la côte.
        
        Args:
            inundation_freq: Fréquence d'inondation
            distance_coast: Distance à la côte
            
        Returns:
            Proxy de salinité (0-1, où 1 = eau salée)
        """
        # Normaliser la distance
        max_dist = np.max(distance_coast)
        normalized_distance = distance_coast / max_dist if max_dist > 0 else distance_coast
        
        # Salinité élevée près de la côte et avec forte inundation
        salinity_proxy = (
            inundation_freq * 0.7 +           # Influence de l'inondation
            (1 - normalized_distance) * 0.3   # Influence de la proximité côtière
        )
        
        # Ajouter de la variabilité spatiale
        noise = np.random.normal(0, 0.05, salinity_proxy.shape)
        salinity_proxy += noise
        
        return np.clip(salinity_proxy, 0, 1).astype(np.float32)
    
    def _create_elevation_map(self, shape: Tuple[int, int], 
                            distance_coast: np.ndarray) -> np.ndarray:
        """
        Crée une carte d'élévation simplifiée.
        
        Args:
            shape: Forme de la carte
            distance_coast: Distance à la côte
            
        Returns:
            Carte d'élévation (mètres au-dessus du niveau de la mer)
        """
        # Élévation basée sur la distance à la côte avec variation
        max_dist = np.max(distance_coast)
        normalized_distance = distance_coast / max_dist if max_dist > 0 else distance_coast
        
        # Élévation croissante avec la distance (0-5m)
        elevation = normalized_distance * 5.0
        
        # Ajouter de la variabilité topographique
        noise = np.random.normal(0, 0.5, elevation.shape)
        elevation += noise
        
        # Contraindre dans une plage réaliste pour les mangroves
        return np.clip(elevation, 0, 8).astype(np.float32)
    
    def classify_mangrove_types(self, hydrodynamic_maps: Dict) -> Dict:
        """
        Classifie les types de mangroves selon les conditions hydrodynamiques.
        
        Args:
            hydrodynamic_maps: Cartes hydrodynamiques
            
        Returns:
            Résultats de classification
        """
        logger.info("Classification des types de mangroves...")
        
        classification_results = self.mangrove_classifier.classify_mangrove_map(
            inundation_frequency=hydrodynamic_maps['inundation_frequency'],
            tidal_range=hydrodynamic_maps['tidal_range'],
            distance_to_coast=hydrodynamic_maps['distance_to_coast'],
            salinity_proxy=hydrodynamic_maps['salinity_proxy'],
            elevation=hydrodynamic_maps['elevation']
        )
        
        self.results['mangrove_classification'] = classification_results
        logger.info("Classification terminée")
        
        return classification_results
    
    def analyze_carbon_dynamics(self, 
                              mangrove_classification: Dict,
                              hydrodynamic_maps: Dict,
                              climate_scenarios: Optional[Dict] = None) -> Dict:
        """
        Analyse les dynamiques de carbone.
        
        Args:
            mangrove_classification: Résultats de classification
            hydrodynamic_maps: Cartes hydrodynamiques
            climate_scenarios: Scénarios climatiques optionnels
            
        Returns:
            Résultats d'analyse du carbone
        """
        logger.info("Analyse des dynamiques de carbone...")
        
        prediction_map = mangrove_classification['prediction_map']
        
        # Calculer la séquestration de base
        base_sequestration = self.carbon_analyzer.calculate_base_sequestration(
            prediction_map, self.area_per_pixel_ha
        )
        
        # Appliquer les ajustements environnementaux
        adjusted_sequestration = self.carbon_analyzer.apply_environmental_adjustments(
            base_sequestration,
            salinity_map=hydrodynamic_maps['salinity_proxy'] * 35,  # Conversion en ppt
            tidal_range_map=hydrodynamic_maps['tidal_range'],
            inundation_freq_map=hydrodynamic_maps['inundation_frequency']
        )
        
        # Calculer les stocks de carbone
        carbon_stocks = self.carbon_analyzer.calculate_carbon_stocks(
            adjusted_sequestration
        )
        
        # Évaluer la vulnérabilité (scénarios par défaut si non fournis)
        if climate_scenarios is None:
            climate_scenarios = {
                'sea_level_rise': 0.3,      # 30 cm
                'temperature_increase': 2.0, # 2°C
                'storm_frequency': 1.2      # 20% d'augmentation
            }
        
        vulnerability = self.carbon_analyzer.assess_carbon_vulnerability(
            carbon_stocks, **climate_scenarios
        )
        
        # Générer le rapport complet
        carbon_report = self.carbon_analyzer.generate_carbon_report(
            prediction_map,
            adjusted_sequestration,
            carbon_stocks,
            vulnerability,
            self.area_per_pixel_ha
        )
        
        carbon_results = {
            'base_sequestration': base_sequestration,
            'adjusted_sequestration': adjusted_sequestration,
            'carbon_stocks': carbon_stocks,
            'vulnerability_assessment': vulnerability,
            'carbon_report': carbon_report
        }
        
        self.results['carbon_analysis'] = carbon_results
        logger.info("Analyse du carbone terminée")
        
        return carbon_results
    
    def correlate_tides_and_carbon(self, processed_data: Dict) -> Dict:
        """
        Analyse les corrélations entre marées et indices de végétation.
        
        Args:
            processed_data: Données traitées
            
        Returns:
            Résultats de corrélation
        """
        logger.info("Analyse des corrélations marées-carbone...")
        
        tidal_data = processed_data['tidal_data']
        temporal_patterns = processed_data['temporal_patterns']
        
        correlations = {}
        
        # Corrélations avec les indices spectraux si disponibles
        if 'timestamps' in temporal_patterns:
            for index_name in ['ndvi', 'ndwi', 'evi']:
                if index_name in temporal_patterns and temporal_patterns[index_name]:
                    correlation_result = self.tidal_analyzer.correlate_with_satellite_data(
                        tidal_data,
                        temporal_patterns['timestamps'],
                        temporal_patterns[index_name],
                        index_name.upper()
                    )
                    correlations[index_name] = correlation_result
        
        # Analyser les patterns saisonniers
        seasonal_patterns = self.tidal_analyzer.analyze_seasonal_patterns(tidal_data)
        
        correlation_results = {
            'spectral_correlations': correlations,
            'seasonal_patterns': seasonal_patterns
        }
        
        self.results['tidal_correlations'] = correlation_results
        logger.info("Analyse des corrélations terminée")
        
        return correlation_results
    
    def run_complete_analysis(self, 
                            image_paths: List[str],
                            timestamps: List[datetime],
                            start_date: datetime,
                            end_date: datetime,
                            climate_scenarios: Optional[Dict] = None) -> Dict:
        """
        Exécute l'analyse complète du modèle hydrodynamique.
        
        Args:
            image_paths: Chemins vers les images satellites
            timestamps: Timestamps des images
            start_date: Date de début d'analyse
            end_date: Date de fin d'analyse
            climate_scenarios: Scénarios climatiques optionnels
            
        Returns:
            Résultats complets de l'analyse
        """
        logger.info("=== DÉBUT DE L'ANALYSE HYDRODYNAMIQUE COMPLÈTE ===")
        
        try:
            # 1. Traitement des données
            processed_data = self.load_and_process_data(
                image_paths, timestamps, start_date, end_date
            )
            
            # 2. Création des cartes hydrodynamiques
            hydrodynamic_maps = self.create_hydrodynamic_maps(processed_data)
            
            # 3. Classification des types de mangroves
            mangrove_classification = self.classify_mangrove_types(hydrodynamic_maps)
            
            # 4. Analyse des dynamiques de carbone
            carbon_analysis = self.analyze_carbon_dynamics(
                mangrove_classification, hydrodynamic_maps, climate_scenarios
            )
            
            # 5. Corrélations marées-carbone
            tidal_correlations = self.correlate_tides_and_carbon(processed_data)
            
            # Compiler les résultats finaux
            complete_results = {
                'metadata': {
                    'location': {'lat': self.location_lat, 'lon': self.location_lon},
                    'pixel_size_m': self.pixel_size_m,
                    'area_per_pixel_ha': self.area_per_pixel_ha,
                    'analysis_period': {'start': start_date, 'end': end_date},
                    'climate_scenarios': climate_scenarios
                },
                'processed_data': processed_data,
                'hydrodynamic_maps': hydrodynamic_maps,
                'mangrove_classification': mangrove_classification,
                'carbon_analysis': carbon_analysis,
                'tidal_correlations': tidal_correlations
            }
            
            self.results['complete_analysis'] = complete_results
            
            logger.info("=== ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS ===")
            return complete_results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse complète: {e}")
            raise
    
    def export_results(self, output_dir: str, prefix: str = "hydrodynamic_analysis"):
        """
        Exporte tous les résultats de l'analyse.
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Export des résultats vers: {output_dir}")
        
        if 'complete_analysis' not in self.results:
            logger.error("Aucune analyse complète trouvée à exporter")
            return
        
        results = self.results['complete_analysis']
        base_path = os.path.join(output_dir, prefix)
        
        # Exporter les cartes hydrodynamiques
        hydro_maps = results['hydrodynamic_maps']
        hydro_path = f"{base_path}_hydrodynamic_maps.npz"
        np.savez_compressed(hydro_path, **hydro_maps)
        
        # Exporter la classification des mangroves
        self.mangrove_classifier.export_classification_results(
            results['mangrove_classification'], f"{base_path}_mangrove"
        )
        
        # Exporter l'analyse du carbone
        carbon_maps = {
            'base_sequestration': results['carbon_analysis']['base_sequestration'],
            'adjusted_sequestration': results['carbon_analysis']['adjusted_sequestration'],
            'soil_carbon_stock': results['carbon_analysis']['carbon_stocks']['soil_carbon_stock'],
            'biomass_carbon_stock': results['carbon_analysis']['carbon_stocks']['biomass_carbon_stock'],
            'total_carbon_stock': results['carbon_analysis']['carbon_stocks']['total_carbon_stock']
        }
        
        self.carbon_analyzer.export_carbon_analysis(
            results['carbon_analysis']['carbon_report'],
            carbon_maps,
            f"{base_path}_carbon"
        )
        
        # Exporter l'analyse des marées
        self.tidal_analyzer.export_tidal_analysis(
            results['processed_data']['tidal_data'],
            results['tidal_correlations']['spectral_correlations'],
            results['tidal_correlations']['seasonal_patterns'],
            f"{base_path}_tidal"
        )
        
        # Exporter un résumé global
        summary = self._create_analysis_summary(results)
        summary_path = f"{base_path}_summary.json"
        
        def json_serializer(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=json_serializer)
        
        logger.info("Export terminé:")
        logger.info(f"  - Cartes hydrodynamiques: {hydro_path}")
        logger.info(f"  - Classification mangroves: {base_path}_mangrove_*")
        logger.info(f"  - Analyse carbone: {base_path}_carbon_*")
        logger.info(f"  - Analyse marées: {base_path}_tidal_*")
        logger.info(f"  - Résumé: {summary_path}")
    
    def _create_analysis_summary(self, results: Dict) -> Dict:
        """
        Crée un résumé de l'analyse complète.
        
        Args:
            results: Résultats de l'analyse
            
        Returns:
            Dictionnaire de résumé
        """
        summary = {
            'metadata': results['metadata'],
            'key_findings': {}
        }
        
        # Résumé de la classification
        if 'mangrove_classification' in results:
            summary['key_findings']['mangrove_types'] = results['mangrove_classification']['type_statistics']
        
        # Résumé du carbone
        if 'carbon_analysis' in results:
            carbon_stats = results['carbon_analysis']['carbon_report']['global_statistics']
            summary['key_findings']['carbon_dynamics'] = {
                'total_area_ha': carbon_stats['total_area_ha'],
                'annual_sequestration_MgC': carbon_stats['total_annual_sequestration_MgC'],
                'total_stock_MgC': carbon_stats['total_carbon_stock_MgC'],
                'mean_sequestration_rate': carbon_stats['mean_sequestration_rate_MgC_ha_yr']
            }
            
            vulnerability_stats = results['carbon_analysis']['carbon_report']['vulnerability_assessment']
            summary['key_findings']['vulnerability'] = vulnerability_stats
        
        # Résumé des corrélations
        if 'tidal_correlations' in results:
            correlations = results['tidal_correlations']['spectral_correlations']
            if correlations:
                summary['key_findings']['tidal_correlations'] = {
                    index: corr.get('pearson_correlation', 'N/A')
                    for index, corr in correlations.items()
                    if isinstance(corr, dict)
                }
        
        return summary
