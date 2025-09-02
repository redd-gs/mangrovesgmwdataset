"""
Analyseur de séquestration de carbone dans les mangroves.

Ce module modélise les taux de séquestration de carbone en fonction du type de mangrove
et des conditions hydrodynamiques, en se basant sur la littérature scientifique.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


class CarbonSequestrationAnalyzer:
    """
    Classe pour analyser la séquestration de carbone dans les mangroves.
    """
    
    def __init__(self):
        """
        Initialise l'analyseur de séquestration de carbone.
        """
        # Taux de séquestration de carbone basés sur la littérature
        # (Mg C ha⁻¹ an⁻¹)
        self.base_sequestration_rates = {
            'marine': {
                'mean': 8.5,      # Taux élevé dû à l'apport constant de sédiments
                'std': 2.1,
                'range': (4.2, 15.3)
            },
            'estuarine': {
                'mean': 12.8,     # Taux le plus élevé (apport nutriments + sédiments)
                'std': 3.5,
                'range': (6.8, 22.1)
            },
            'terrestrial': {
                'mean': 4.2,      # Taux plus faible (moins d'apport sédimentaire)
                'std': 1.8,
                'range': (1.5, 8.9)
            }
        }
        
        # Facteurs d'ajustement selon les conditions environnementales
        self.environmental_factors = {
            'salinity': {
                'optimal_range': (15, 25),  # ppt
                'factor_range': (0.7, 1.3)
            },
            'temperature': {
                'optimal_range': (25, 30),  # °C
                'factor_range': (0.8, 1.2)
            },
            'tidal_range': {
                'optimal_range': (1.0, 3.0),  # m
                'factor_range': (0.6, 1.4)
            },
            'inundation_frequency': {
                'optimal_range': (0.3, 0.7),  # fraction
                'factor_range': (0.5, 1.5)
            }
        }
    
    def calculate_base_sequestration(self, mangrove_type_map: np.ndarray, 
                                   area_per_pixel: float) -> np.ndarray:
        """
        Calcule la séquestration de base selon le type de mangrove.
        
        Args:
            mangrove_type_map: Carte des types de mangroves (0: marine, 1: estuarine, 2: terrestrial)
            area_per_pixel: Surface par pixel en hectares
            
        Returns:
            Carte de séquestration de base (Mg C an⁻¹)
        """
        sequestration_map = np.zeros_like(mangrove_type_map, dtype=np.float32)
        
        # Types de mangroves : 0=marine, 1=estuarine, 2=terrestrial
        type_names = ['marine', 'estuarine', 'terrestrial']
        
        for type_id, type_name in enumerate(type_names):
            mask = mangrove_type_map == type_id
            if np.any(mask):
                # Utiliser une distribution normale pour la variabilité
                base_rate = self.base_sequestration_rates[type_name]['mean']
                std_rate = self.base_sequestration_rates[type_name]['std']
                
                # Générer des taux avec variabilité spatiale
                n_pixels = np.sum(mask)
                rates = np.random.normal(base_rate, std_rate, n_pixels)
                
                # Contraindre dans la plage observée
                min_rate, max_rate = self.base_sequestration_rates[type_name]['range']
                rates = np.clip(rates, min_rate, max_rate)
                
                # Convertir en séquestration totale par pixel
                sequestration_map[mask] = rates * area_per_pixel
        
        return sequestration_map
    
    def apply_environmental_adjustments(self, 
                                      base_sequestration: np.ndarray,
                                      salinity_map: Optional[np.ndarray] = None,
                                      temperature_map: Optional[np.ndarray] = None,
                                      tidal_range_map: Optional[np.ndarray] = None,
                                      inundation_freq_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applique les ajustements environnementaux aux taux de séquestration.
        
        Args:
            base_sequestration: Carte de séquestration de base
            salinity_map: Carte de salinité (ppt)
            temperature_map: Carte de température (°C)
            tidal_range_map: Carte d'amplitude de marée (m)
            inundation_freq_map: Carte de fréquence d'inondation (0-1)
            
        Returns:
            Carte de séquestration ajustée
        """
        adjusted_sequestration = base_sequestration.copy()
        
        environmental_maps = {
            'salinity': salinity_map,
            'temperature': temperature_map,
            'tidal_range': tidal_range_map,
            'inundation_frequency': inundation_freq_map
        }
        
        for factor_name, env_map in environmental_maps.items():
            if env_map is not None:
                factor_map = self._calculate_environmental_factor(env_map, factor_name)
                adjusted_sequestration *= factor_map
        
        return adjusted_sequestration
    
    def _calculate_environmental_factor(self, env_map: np.ndarray, 
                                      factor_name: str) -> np.ndarray:
        """
        Calcule le facteur d'ajustement environnemental.
        
        Args:
            env_map: Carte de la variable environnementale
            factor_name: Nom du facteur environnemental
            
        Returns:
            Carte des facteurs d'ajustement
        """
        factor_config = self.environmental_factors[factor_name]
        optimal_min, optimal_max = factor_config['optimal_range']
        factor_min, factor_max = factor_config['factor_range']
        
        factor_map = np.ones_like(env_map, dtype=np.float32)
        
        # Facteur optimal dans la plage optimale
        optimal_mask = (env_map >= optimal_min) & (env_map <= optimal_max)
        factor_map[optimal_mask] = np.random.uniform(0.95, 1.05, np.sum(optimal_mask))
        
        # Facteur réduit en dehors de la plage optimale
        below_optimal = env_map < optimal_min
        above_optimal = env_map > optimal_max
        
        if np.any(below_optimal):
            # Interpolation linéaire pour les valeurs faibles
            min_env = np.min(env_map[below_optimal])
            factor_below = np.interp(
                env_map[below_optimal],
                [min_env, optimal_min],
                [factor_min, 1.0]
            )
            factor_map[below_optimal] = factor_below
        
        if np.any(above_optimal):
            # Interpolation linéaire pour les valeurs élevées
            max_env = np.max(env_map[above_optimal])
            factor_above = np.interp(
                env_map[above_optimal],
                [optimal_max, max_env],
                [1.0, factor_min]
            )
            factor_map[above_optimal] = factor_above
        
        return factor_map
    
    def model_temporal_variability(self, 
                                 sequestration_map: np.ndarray,
                                 tidal_data: pd.DataFrame,
                                 time_periods: int = 12) -> Dict:
        """
        Modélise la variabilité temporelle de la séquestration.
        
        Args:
            sequestration_map: Carte de séquestration annuelle
            tidal_data: Données de marée
            time_periods: Nombre de périodes temporelles à modéliser
            
        Returns:
            Dictionnaire avec les variations temporelles
        """
        # Calculer les variations saisonnières basées sur les marées
        temporal_variations = {}
        
        # Diviser l'année en périodes
        period_length = 365 // time_periods
        
        for period in range(time_periods):
            start_day = period * period_length
            end_day = min((period + 1) * period_length, 365)
            
            # Simuler les conditions pour cette période
            period_factor = self._calculate_seasonal_factor(
                period, time_periods, tidal_data
            )
            
            period_sequestration = sequestration_map * period_factor
            
            temporal_variations[f'period_{period+1}'] = {
                'days': (start_day, end_day),
                'sequestration_map': period_sequestration,
                'total_sequestration': np.sum(period_sequestration),
                'mean_rate': np.mean(period_sequestration),
                'seasonal_factor': period_factor
            }
        
        return temporal_variations
    
    def _calculate_seasonal_factor(self, period: int, total_periods: int, 
                                 tidal_data: pd.DataFrame) -> float:
        """
        Calcule le facteur saisonnier pour une période donnée.
        
        Args:
            period: Période actuelle (0-indexed)
            total_periods: Nombre total de périodes
            tidal_data: Données de marée
            
        Returns:
            Facteur saisonnier
        """
        # Calculer la phase saisonnière (0 à 2π)
        seasonal_phase = 2 * np.pi * period / total_periods
        
        # Variation saisonnière basée sur les marées moyennes
        if not tidal_data.empty and 'tide_height' in tidal_data.columns:
            # Estimer la variabilité saisonnière des marées
            tidal_variability = tidal_data['tide_height'].std()
            base_factor = 0.8 + 0.4 * np.sin(seasonal_phase)  # Variation de 0.8 à 1.2
            
            # Ajuster selon la variabilité des marées
            tidal_adjustment = 1 + 0.1 * (tidal_variability / tidal_data['tide_height'].mean())
            seasonal_factor = base_factor * tidal_adjustment
        else:
            # Facteur saisonnier par défaut
            seasonal_factor = 0.9 + 0.2 * np.sin(seasonal_phase)
        
        return np.clip(seasonal_factor, 0.5, 1.5)
    
    def calculate_carbon_stocks(self, 
                              annual_sequestration: np.ndarray,
                              mangrove_age_map: Optional[np.ndarray] = None,
                              soil_depth: float = 1.0) -> Dict:
        """
        Calcule les stocks de carbone dans le sol et la biomasse.
        
        Args:
            annual_sequestration: Séquestration annuelle (Mg C an⁻¹)
            mangrove_age_map: Âge des mangroves (années)
            soil_depth: Profondeur de sol considérée (mètres)
            
        Returns:
            Dictionnaire des stocks de carbone
        """
        # Si pas de carte d'âge, utiliser un âge moyen
        if mangrove_age_map is None:
            mangrove_age_map = np.full_like(annual_sequestration, 25.0)  # 25 ans par défaut
        
        # Calculer les stocks selon la profondeur
        # Facteur de décroissance exponentielle avec la profondeur
        depth_factor = np.exp(-soil_depth / 0.5)  # Décroissance caractéristique à 0.5m
        
        # Stock de carbone dans le sol (accumulation sur l'âge)
        # Avec saturation progressive
        saturation_factor = 1 - np.exp(-mangrove_age_map / 50)  # Saturation à ~50 ans
        soil_carbon_stock = annual_sequestration * mangrove_age_map * saturation_factor * depth_factor
        
        # Stock de carbone dans la biomasse (relation allométrique simplifiée)
        # Biomasse = f(âge, productivité)
        biomass_factor = np.sqrt(mangrove_age_map) / 10  # Facteur d'âge
        productivity_factor = annual_sequestration / np.mean(annual_sequestration)
        biomass_carbon_stock = annual_sequestration * biomass_factor * productivity_factor * 5  # Facteur de conversion
        
        # Stock total
        total_carbon_stock = soil_carbon_stock + biomass_carbon_stock
        
        return {
            'soil_carbon_stock': soil_carbon_stock,
            'biomass_carbon_stock': biomass_carbon_stock,
            'total_carbon_stock': total_carbon_stock,
            'soil_percentage': (np.sum(soil_carbon_stock) / np.sum(total_carbon_stock)) * 100,
            'biomass_percentage': (np.sum(biomass_carbon_stock) / np.sum(total_carbon_stock)) * 100
        }
    
    def assess_carbon_vulnerability(self, 
                                  carbon_stocks: Dict,
                                  sea_level_rise: float = 0.0,
                                  temperature_increase: float = 0.0,
                                  storm_frequency: float = 0.0) -> Dict:
        """
        Évalue la vulnérabilité des stocks de carbone aux changements climatiques.
        
        Args:
            carbon_stocks: Stocks de carbone calculés
            sea_level_rise: Élévation du niveau de la mer (m)
            temperature_increase: Augmentation de température (°C)
            storm_frequency: Fréquence des tempêtes (facteur multiplicatif)
            
        Returns:
            Dictionnaire d'évaluation de vulnérabilité
        """
        vulnerability_assessment = {}
        
        # Facteur de risque d'érosion (montée du niveau de la mer)
        erosion_risk = np.tanh(sea_level_rise / 0.5)  # Risque saturé à 0.5m
        
        # Facteur de stress thermique
        thermal_stress = np.tanh(temperature_increase / 3.0)  # Stress saturé à 3°C
        
        # Facteur de perturbation par tempêtes
        storm_disturbance = np.tanh(storm_frequency)
        
        # Vulnérabilité combinée
        combined_vulnerability = (erosion_risk + thermal_stress + storm_disturbance) / 3
        
        # Pertes potentielles de carbone
        total_stock = carbon_stocks['total_carbon_stock']
        
        # Perte de carbone du sol (plus vulnérable à l'érosion)
        soil_loss_factor = erosion_risk * 0.3 + storm_disturbance * 0.2
        potential_soil_loss = carbon_stocks['soil_carbon_stock'] * soil_loss_factor
        
        # Perte de carbone de biomasse (plus vulnérable au stress thermique)
        biomass_loss_factor = thermal_stress * 0.4 + storm_disturbance * 0.3
        potential_biomass_loss = carbon_stocks['biomass_carbon_stock'] * biomass_loss_factor
        
        total_potential_loss = potential_soil_loss + potential_biomass_loss
        
        vulnerability_assessment = {
            'erosion_risk': erosion_risk,
            'thermal_stress': thermal_stress,
            'storm_disturbance': storm_disturbance,
            'combined_vulnerability': combined_vulnerability,
            'potential_soil_loss': potential_soil_loss,
            'potential_biomass_loss': potential_biomass_loss,
            'total_potential_loss': total_potential_loss,
            'loss_percentage': (np.sum(total_potential_loss) / np.sum(total_stock)) * 100,
            'vulnerable_areas': combined_vulnerability > 0.5  # Seuil de vulnérabilité élevée
        }
        
        return vulnerability_assessment
    
    def generate_carbon_report(self, 
                             mangrove_types: np.ndarray,
                             sequestration_map: np.ndarray,
                             carbon_stocks: Dict,
                             vulnerability: Dict,
                             area_per_pixel: float) -> Dict:
        """
        Génère un rapport complet sur le carbone.
        
        Args:
            mangrove_types: Carte des types de mangroves
            sequestration_map: Carte de séquestration
            carbon_stocks: Stocks de carbone
            vulnerability: Évaluation de vulnérabilité
            area_per_pixel: Surface par pixel (ha)
            
        Returns:
            Rapport complet
        """
        total_area = np.sum(mangrove_types >= 0) * area_per_pixel
        
        # Statistiques par type de mangrove
        type_stats = {}
        type_names = {0: 'marine', 1: 'estuarine', 2: 'terrestrial'}
        
        for type_id, type_name in type_names.items():
            mask = mangrove_types == type_id
            if np.any(mask):
                type_area = np.sum(mask) * area_per_pixel
                type_sequestration = np.sum(sequestration_map[mask])
                type_stock = np.sum(carbon_stocks['total_carbon_stock'][mask])
                
                type_stats[type_name] = {
                    'area_ha': float(type_area),
                    'area_percentage': float((type_area / total_area) * 100),
                    'annual_sequestration_MgC': float(type_sequestration),
                    'total_stock_MgC': float(type_stock),
                    'sequestration_rate_MgC_ha_yr': float(type_sequestration / type_area) if type_area > 0 else 0,
                    'stock_density_MgC_ha': float(type_stock / type_area) if type_area > 0 else 0
                }
        
        # Statistiques globales
        global_stats = {
            'total_area_ha': float(total_area),
            'total_annual_sequestration_MgC': float(np.sum(sequestration_map)),
            'total_carbon_stock_MgC': float(np.sum(carbon_stocks['total_carbon_stock'])),
            'mean_sequestration_rate_MgC_ha_yr': float(np.sum(sequestration_map) / total_area) if total_area > 0 else 0,
            'mean_stock_density_MgC_ha': float(np.sum(carbon_stocks['total_carbon_stock']) / total_area) if total_area > 0 else 0,
            'soil_carbon_percentage': float(carbon_stocks['soil_percentage']),
            'biomass_carbon_percentage': float(carbon_stocks['biomass_percentage'])
        }
        
        # Vulnérabilité
        vulnerability_stats = {
            'mean_vulnerability': float(np.mean(vulnerability['combined_vulnerability'])),
            'high_vulnerability_area_ha': float(np.sum(vulnerability['vulnerable_areas']) * area_per_pixel),
            'potential_carbon_loss_MgC': float(np.sum(vulnerability['total_potential_loss'])),
            'carbon_loss_percentage': float(vulnerability['loss_percentage'])
        }
        
        return {
            'global_statistics': global_stats,
            'type_statistics': type_stats,
            'vulnerability_assessment': vulnerability_stats,
            'methodology': {
                'base_sequestration_rates': self.base_sequestration_rates,
                'environmental_factors': self.environmental_factors
            }
        }
    
    def export_carbon_analysis(self, report: Dict, carbon_maps: Dict, 
                             output_path: str):
        """
        Exporte l'analyse complète du carbone.
        
        Args:
            report: Rapport d'analyse
            carbon_maps: Cartes de carbone
            output_path: Chemin de base pour les fichiers
        """
        import json
        
        # Exporter le rapport en JSON
        report_path = f"{output_path}_carbon_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Exporter les cartes en format numpy
        maps_path = f"{output_path}_carbon_maps.npz"
        np.savez_compressed(maps_path, **carbon_maps)
        
        logger.info(f"Analyse du carbone exportée:")
        logger.info(f"  - Rapport: {report_path}")
        logger.info(f"  - Cartes: {maps_path}")
        
        # Afficher un résumé
        global_stats = report['global_statistics']
        logger.info("Résumé de l'analyse du carbone:")
        logger.info(f"  - Surface totale: {global_stats['total_area_ha']:.1f} ha")
        logger.info(f"  - Séquestration annuelle: {global_stats['total_annual_sequestration_MgC']:.1f} Mg C/an")
        logger.info(f"  - Stock total: {global_stats['total_carbon_stock_MgC']:.1f} Mg C")
        logger.info(f"  - Taux moyen: {global_stats['mean_sequestration_rate_MgC_ha_yr']:.2f} Mg C/ha/an")
