"""
Analyseur de marées pour l'étude des mangroves.

Ce module analyse les cycles de marée et leur impact sur les écosystèmes de mangroves,
en utilisant des données de marée et des observations satellitaires.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy import signal
from scipy.interpolate import interp1d
import requests
import json

logger = logging.getLogger(__name__)


class TidalAnalyzer:
    """
    Classe pour analyser les données de marée et leur impact sur les mangroves.
    """
    
    def __init__(self, location_lat: float, location_lon: float):
        """
        Initialise l'analyseur de marées.
        
        Args:
            location_lat: Latitude de la zone d'étude
            location_lon: Longitude de la zone d'étude
        """
        self.location_lat = location_lat
        self.location_lon = location_lon
        self.tidal_data = None
        
    def get_tidal_data_theoretical(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Génère des données de marée théoriques basées sur des harmoniques.
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données de marée
        """
        # Créer une série temporelle
        time_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Composantes harmoniques principales des marées
        # M2: semi-diurne principale (12.42h)
        # S2: semi-diurne solaire (12h)
        # N2: semi-diurne lunaire (12.66h)
        # K1: diurne luni-solaire (23.93h)
        # O1: diurne lunaire (25.82h)
        
        hours_since_start = np.array([(t - time_range[0]).total_seconds() / 3600 
                                     for t in time_range])
        
        # Amplitudes (en mètres) - ajustées selon la région
        amp_M2 = 1.5  # Amplitude principale
        amp_S2 = 0.5  # Amplitude solaire
        amp_N2 = 0.3  # Amplitude lunaire
        amp_K1 = 0.4  # Amplitude diurne
        amp_O1 = 0.2  # Amplitude diurne lunaire
        
        # Périodes (en heures)
        period_M2 = 12.42
        period_S2 = 12.00
        period_N2 = 12.66
        period_K1 = 23.93
        period_O1 = 25.82
        
        # Calcul des hauteurs de marée
        tide_height = (
            amp_M2 * np.sin(2 * np.pi * hours_since_start / period_M2) +
            amp_S2 * np.sin(2 * np.pi * hours_since_start / period_S2) +
            amp_N2 * np.sin(2 * np.pi * hours_since_start / period_N2) +
            amp_K1 * np.sin(2 * np.pi * hours_since_start / period_K1) +
            amp_O1 * np.sin(2 * np.pi * hours_since_start / period_O1)
        )
        
        # Ajouter du bruit réaliste
        noise = np.random.normal(0, 0.1, len(tide_height))
        tide_height += noise
        
        # Créer le DataFrame
        tidal_df = pd.DataFrame({
            'timestamp': time_range,
            'tide_height': tide_height,
            'tide_velocity': np.gradient(tide_height),  # Vitesse de marée
        })
        
        return tidal_df
    
    def classify_tidal_conditions(self, tidal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classifie les conditions de marée.
        
        Args:
            tidal_data: Données de marée
            
        Returns:
            DataFrame avec classification des marées
        """
        df = tidal_data.copy()
        
        # Percentiles pour classification
        high_tide_threshold = np.percentile(df['tide_height'], 75)
        low_tide_threshold = np.percentile(df['tide_height'], 25)
        
        # Classification des conditions
        conditions = []
        phases = []
        
        for i, row in df.iterrows():
            height = row['tide_height']
            velocity = row['tide_velocity']
            
            # Classification par hauteur
            if height > high_tide_threshold:
                condition = 'high_tide'
            elif height < low_tide_threshold:
                condition = 'low_tide'
            else:
                condition = 'mid_tide'
            
            # Classification par phase (montante/descendante)
            if velocity > 0.05:
                phase = 'rising'
            elif velocity < -0.05:
                phase = 'falling'
            else:
                phase = 'slack'
            
            conditions.append(condition)
            phases.append(phase)
        
        df['tidal_condition'] = conditions
        df['tidal_phase'] = phases
        
        return df
    
    def calculate_tidal_range(self, tidal_data: pd.DataFrame, 
                            window_hours: int = 24) -> pd.DataFrame:
        """
        Calcule l'amplitude de marée sur une fenêtre glissante.
        
        Args:
            tidal_data: Données de marée
            window_hours: Taille de la fenêtre en heures
            
        Returns:
            DataFrame avec amplitude de marée
        """
        df = tidal_data.copy()
        df = df.set_index('timestamp')
        
        # Fenêtre glissante pour calculer l'amplitude
        window = f'{window_hours}h'
        tidal_range = df['tide_height'].rolling(window=window).apply(
            lambda x: x.max() - x.min()
        )
        
        df['tidal_range'] = tidal_range
        
        # Classification du type de marée
        range_conditions = []
        for range_val in tidal_range:
            if pd.isna(range_val):
                range_conditions.append('unknown')
            elif range_val > 4.0:
                range_conditions.append('macro_tidal')  # > 4m
            elif range_val > 2.0:
                range_conditions.append('meso_tidal')   # 2-4m
            else:
                range_conditions.append('micro_tidal')  # < 2m
        
        df['tidal_range_type'] = range_conditions
        
        return df.reset_index()
    
    def identify_extreme_events(self, tidal_data: pd.DataFrame, 
                              threshold_std: float = 2.0) -> pd.DataFrame:
        """
        Identifie les événements de marée extrêmes.
        
        Args:
            tidal_data: Données de marée
            threshold_std: Seuil en écarts-types pour les événements extrêmes
            
        Returns:
            DataFrame des événements extrêmes
        """
        df = tidal_data.copy()
        
        # Calculer les statistiques
        mean_height = df['tide_height'].mean()
        std_height = df['tide_height'].std()
        
        # Identifier les événements extrêmes
        extreme_high = df['tide_height'] > (mean_height + threshold_std * std_height)
        extreme_low = df['tide_height'] < (mean_height - threshold_std * std_height)
        
        # Filtrer les événements extrêmes
        extreme_events = df[extreme_high | extreme_low].copy()
        extreme_events['event_type'] = np.where(
            extreme_events['tide_height'] > mean_height, 
            'extreme_high', 
            'extreme_low'
        )
        
        return extreme_events
    
    def correlate_with_satellite_data(self, tidal_data: pd.DataFrame, 
                                    satellite_timestamps: List[datetime],
                                    satellite_values: List[float],
                                    index_name: str = 'NDVI') -> Dict:
        """
        Corrèle les données de marée avec les observations satellitaires.
        
        Args:
            tidal_data: Données de marée
            satellite_timestamps: Timestamps des observations satellites
            satellite_values: Valeurs des indices satellitaires
            index_name: Nom de l'indice satellitaire
            
        Returns:
            Dictionnaire des résultats de corrélation
        """
        # Interpoler les données de marée aux timestamps satellites
        tidal_interp = interp1d(
            [t.timestamp() for t in tidal_data['timestamp']], 
            tidal_data['tide_height'],
            kind='linear',
            fill_value='extrapolate'
        )
        
        # Obtenir les hauteurs de marée aux moments des observations satellites
        tidal_heights_at_sat = tidal_interp([t.timestamp() for t in satellite_timestamps])
        
        # Calculer les corrélations
        correlation_results = {
            'timestamps': satellite_timestamps,
            'tidal_heights': tidal_heights_at_sat,
            'satellite_values': satellite_values,
            'index_name': index_name
        }
        
        # Corrélation de Pearson
        if len(satellite_values) > 2:
            correlation_coef = np.corrcoef(tidal_heights_at_sat, satellite_values)[0, 1]
            correlation_results['pearson_correlation'] = correlation_coef
            
            # Analyse par phase de marée
            tidal_phases = []
            for height in tidal_heights_at_sat:
                if height > np.percentile(tidal_heights_at_sat, 75):
                    tidal_phases.append('high')
                elif height < np.percentile(tidal_heights_at_sat, 25):
                    tidal_phases.append('low')
                else:
                    tidal_phases.append('mid')
            
            correlation_results['tidal_phases'] = tidal_phases
            
            # Moyennes par phase
            phase_means = {}
            for phase in ['high', 'mid', 'low']:
                phase_indices = [i for i, p in enumerate(tidal_phases) if p == phase]
                if phase_indices:
                    phase_values = [satellite_values[i] for i in phase_indices]
                    phase_means[f'{phase}_tide_mean_{index_name}'] = np.mean(phase_values)
            
            correlation_results['phase_statistics'] = phase_means
        
        return correlation_results
    
    def analyze_seasonal_patterns(self, tidal_data: pd.DataFrame) -> Dict:
        """
        Analyse les patterns saisonniers des marées.
        
        Args:
            tidal_data: Données de marée
            
        Returns:
            Dictionnaire des patterns saisonniers
        """
        df = tidal_data.copy()
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['timestamp'].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        seasonal_stats = {}
        
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_data = df[df['season'] == season]
            if not season_data.empty:
                seasonal_stats[season] = {
                    'mean_tide_height': season_data['tide_height'].mean(),
                    'std_tide_height': season_data['tide_height'].std(),
                    'max_tide_height': season_data['tide_height'].max(),
                    'min_tide_height': season_data['tide_height'].min(),
                    'mean_tidal_range': season_data.get('tidal_range', pd.Series()).mean()
                }
        
        return seasonal_stats
    
    def export_tidal_analysis(self, tidal_data: pd.DataFrame, 
                            correlations: Dict, 
                            seasonal_patterns: Dict,
                            output_path: str):
        """
        Exporte l'analyse des marées vers des fichiers.
        
        Args:
            tidal_data: Données de marée
            correlations: Résultats de corrélation
            seasonal_patterns: Patterns saisonniers
            output_path: Chemin de base pour les fichiers de sortie
        """
        import os
        
        # Exporter les données de marée
        tidal_csv_path = f"{output_path}_tidal_data.csv"
        tidal_data.to_csv(tidal_csv_path, index=False)
        
        # Exporter les corrélations et patterns en JSON
        analysis_data = {
            'correlations': correlations,
            'seasonal_patterns': seasonal_patterns,
            'location': {
                'latitude': self.location_lat,
                'longitude': self.location_lon
            }
        }
        
        # Convertir les objets non sérialisables
        def convert_for_json(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj
        
        analysis_json_path = f"{output_path}_tidal_analysis.json"
        with open(analysis_json_path, 'w') as f:
            json.dump(convert_for_json(analysis_data), f, indent=2)
        
        logger.info(f"Analyse des marées exportée:")
        logger.info(f"  - Données: {tidal_csv_path}")
        logger.info(f"  - Analyse: {analysis_json_path}")
    
    def predict_optimal_sampling_times(self, tidal_data: pd.DataFrame, 
                                     future_days: int = 30) -> pd.DataFrame:
        """
        Prédit les moments optimaux pour l'échantillonnage en fonction des marées.
        
        Args:
            tidal_data: Données de marée historiques
            future_days: Nombre de jours à prédire
            
        Returns:
            DataFrame des moments optimaux de sampling
        """
        # Extraire les patterns cycliques
        last_date = tidal_data['timestamp'].max()
        future_start = last_date + timedelta(hours=1)
        future_end = future_start + timedelta(days=future_days)
        
        # Générer les prédictions futures
        future_tidal_data = self.get_tidal_data_theoretical(future_start, future_end)
        future_classified = self.classify_tidal_conditions(future_tidal_data)
        
        # Identifier les moments optimaux (marée basse pour accès terrestre, 
        # marée haute pour observation marine)
        optimal_times = []
        
        for condition in ['low_tide', 'high_tide']:
            condition_data = future_classified[
                future_classified['tidal_condition'] == condition
            ]
            
            # Sélectionner un échantillon représentatif
            if not condition_data.empty:
                # Prendre un échantillon tous les 2-3 jours
                sample_interval = max(1, len(condition_data) // (future_days // 2))
                sampled = condition_data.iloc[::sample_interval]
                
                for _, row in sampled.iterrows():
                    optimal_times.append({
                        'timestamp': row['timestamp'],
                        'tide_height': row['tide_height'],
                        'condition': condition,
                        'phase': row['tidal_phase'],
                        'sampling_purpose': 'terrestrial_access' if condition == 'low_tide' else 'marine_observation'
                    })
        
        return pd.DataFrame(optimal_times).sort_values('timestamp')
