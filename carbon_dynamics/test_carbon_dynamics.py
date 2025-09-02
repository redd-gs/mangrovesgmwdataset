"""
Tests unitaires pour le module carbon_dynamics.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_dynamics import (
    TimeSeriesProcessor,
    TidalAnalyzer,
    MangroveTypeClassifier,
    CarbonSequestrationAnalyzer,
    HydrodynamicModel
)


class TestTimeSeriesProcessor:
    """Tests pour TimeSeriesProcessor."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.processor = TimeSeriesProcessor()
        
        # Créer des données d'exemple
        self.sample_image = np.random.rand(4, 50, 50).astype(np.float32)  # 4 bandes
        self.sample_timestamps = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 15),
            datetime(2023, 2, 1)
        ]
        
    def test_extract_spectral_indices(self):
        """Test du calcul des indices spectraux."""
        indices = self.processor.extract_spectral_indices(self.sample_image)
        
        # Vérifier que les indices attendus sont présents
        expected_indices = ['ndvi', 'ndwi', 'evi', 'grvi', 'bg_ratio']
        for index in expected_indices:
            assert index in indices
            assert indices[index].shape == self.sample_image[0].shape
            assert not np.isnan(indices[index]).all()
    
    def test_detect_water_areas(self):
        """Test de la détection des zones d'eau."""
        indices = self.processor.extract_spectral_indices(self.sample_image)
        water_mask = self.processor.detect_water_areas(indices)
        
        assert water_mask.shape == self.sample_image[0].shape
        assert water_mask.dtype == bool
        assert np.any(water_mask) or np.all(~water_mask)  # Soit des zones d'eau, soit aucune
    
    def test_calculate_inundation_frequency(self):
        """Test du calcul de fréquence d'inondation."""
        time_series_data = {
            'timestamps': self.sample_timestamps,
            'images': [self.sample_image for _ in range(3)]
        }
        
        inundation_freq = self.processor.calculate_inundation_frequency(time_series_data)
        
        assert inundation_freq.shape == self.sample_image[0].shape
        assert np.all(inundation_freq >= 0) and np.all(inundation_freq <= 1)


class TestTidalAnalyzer:
    """Tests pour TidalAnalyzer."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.analyzer = TidalAnalyzer(1.3521, 103.8198)  # Singapour
        
    def test_get_tidal_data_theoretical(self):
        """Test de génération de données de marée théoriques."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        tidal_data = self.analyzer.get_tidal_data_theoretical(start_date, end_date)
        
        assert isinstance(tidal_data, pd.DataFrame)
        assert 'timestamp' in tidal_data.columns
        assert 'tide_height' in tidal_data.columns
        assert 'tide_velocity' in tidal_data.columns
        assert len(tidal_data) > 0
        
        # Vérifier la plage des valeurs
        assert tidal_data['tide_height'].std() > 0  # Variation des marées
    
    def test_classify_tidal_conditions(self):
        """Test de classification des conditions de marée."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)  # Une semaine
        
        tidal_data = self.analyzer.get_tidal_data_theoretical(start_date, end_date)
        classified = self.analyzer.classify_tidal_conditions(tidal_data)
        
        assert 'tidal_condition' in classified.columns
        assert 'tidal_phase' in classified.columns
        
        # Vérifier les valeurs attendues
        expected_conditions = {'high_tide', 'mid_tide', 'low_tide'}
        expected_phases = {'rising', 'falling', 'slack'}
        
        assert set(classified['tidal_condition'].unique()).issubset(expected_conditions)
        assert set(classified['tidal_phase'].unique()).issubset(expected_phases)
    
    def test_calculate_tidal_range(self):
        """Test du calcul d'amplitude de marée."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)
        
        tidal_data = self.analyzer.get_tidal_data_theoretical(start_date, end_date)
        classified = self.analyzer.classify_tidal_conditions(tidal_data)
        with_range = self.analyzer.calculate_tidal_range(classified)
        
        assert 'tidal_range' in with_range.columns
        assert 'tidal_range_type' in with_range.columns
        
        # Vérifier que l'amplitude est positive
        valid_ranges = with_range['tidal_range'].dropna()
        if len(valid_ranges) > 0:
            assert np.all(valid_ranges >= 0)


class TestMangroveTypeClassifier:
    """Tests pour MangroveTypeClassifier."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.classifier = MangroveTypeClassifier()
        
        # Créer des données d'exemple
        self.sample_size = (20, 20)
        self.inundation_freq = np.random.rand(*self.sample_size).astype(np.float32)
        self.tidal_range = np.full(self.sample_size, 2.0, dtype=np.float32)
        self.distance_coast = np.random.rand(*self.sample_size).astype(np.float32) * 1000
        self.salinity_proxy = np.random.rand(*self.sample_size).astype(np.float32)
        self.elevation = np.random.rand(*self.sample_size).astype(np.float32) * 3
    
    def test_extract_hydrodynamic_features(self):
        """Test d'extraction des caractéristiques hydrodynamiques."""
        features = self.classifier.extract_hydrodynamic_features(
            self.inundation_freq,
            self.tidal_range,
            self.distance_coast,
            self.salinity_proxy,
            self.elevation
        )
        
        assert features.shape[0] == self.sample_size[0] * self.sample_size[1]
        assert features.shape[1] > 5  # Au moins les 5 features de base + dérivées
        assert not np.isnan(features).all()
    
    def test_create_training_labels(self):
        """Test de création des labels d'entraînement."""
        features = self.classifier.extract_hydrodynamic_features(
            self.inundation_freq,
            self.tidal_range,
            self.distance_coast,
            self.salinity_proxy,
            self.elevation
        )
        
        labels = self.classifier.create_training_labels(features, method='rule_based')
        
        assert len(labels) == features.shape[0]
        assert set(labels).issubset({0, 1, 2})  # Marine, Estuarine, Terrestrial
    
    def test_classify_mangrove_map(self):
        """Test de classification d'une carte complète."""
        results = self.classifier.classify_mangrove_map(
            self.inundation_freq,
            self.tidal_range,
            self.distance_coast,
            self.salinity_proxy,
            self.elevation
        )
        
        assert 'prediction_map' in results
        assert 'probability_maps' in results
        assert 'type_statistics' in results
        assert 'type_mapping' in results
        
        # Vérifier les dimensions
        assert results['prediction_map'].shape == self.sample_size
        
        # Vérifier les statistiques
        total_percentage = sum(stats['percentage'] for stats in results['type_statistics'].values())
        assert abs(total_percentage - 100) < 1e-6  # Doit sommer à 100%


class TestCarbonSequestrationAnalyzer:
    """Tests pour CarbonSequestrationAnalyzer."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.analyzer = CarbonSequestrationAnalyzer()
        
        # Créer une carte de types de mangroves
        self.sample_size = (15, 15)
        self.mangrove_types = np.random.randint(0, 3, self.sample_size)  # 0, 1, 2
        self.area_per_pixel = 0.01  # 0.01 ha par pixel
    
    def test_calculate_base_sequestration(self):
        """Test du calcul de séquestration de base."""
        sequestration = self.analyzer.calculate_base_sequestration(
            self.mangrove_types, 
            self.area_per_pixel
        )
        
        assert sequestration.shape == self.sample_size
        assert np.all(sequestration >= 0)
        
        # Vérifier que les valeurs sont dans des plages raisonnables
        # (dépend des taux de base définis dans la classe)
        assert np.max(sequestration) < 1.0  # Moins de 1 Mg C par petit pixel
    
    def test_apply_environmental_adjustments(self):
        """Test d'application des ajustements environnementaux."""
        base_sequestration = self.analyzer.calculate_base_sequestration(
            self.mangrove_types, 
            self.area_per_pixel
        )
        
        # Créer des cartes environnementales
        salinity_map = np.full(self.sample_size, 20.0, dtype=np.float32)  # 20 ppt
        temperature_map = np.full(self.sample_size, 27.0, dtype=np.float32)  # 27°C
        
        adjusted = self.analyzer.apply_environmental_adjustments(
            base_sequestration,
            salinity_map=salinity_map,
            temperature_map=temperature_map
        )
        
        assert adjusted.shape == base_sequestration.shape
        assert np.all(adjusted >= 0)
        # Les ajustements peuvent augmenter ou diminuer la séquestration
    
    def test_calculate_carbon_stocks(self):
        """Test du calcul des stocks de carbone."""
        annual_sequestration = np.random.rand(*self.sample_size) * 0.1  # Mg C/an
        
        stocks = self.analyzer.calculate_carbon_stocks(annual_sequestration)
        
        required_keys = ['soil_carbon_stock', 'biomass_carbon_stock', 'total_carbon_stock']
        for key in required_keys:
            assert key in stocks
            assert stocks[key].shape == self.sample_size
            assert np.all(stocks[key] >= 0)
        
        # Vérifier que le stock total = sol + biomasse
        total_calculated = stocks['soil_carbon_stock'] + stocks['biomass_carbon_stock']
        np.testing.assert_array_almost_equal(
            stocks['total_carbon_stock'], 
            total_calculated,
            decimal=5
        )
    
    def test_assess_carbon_vulnerability(self):
        """Test d'évaluation de la vulnérabilité."""
        # Créer des stocks de carbone fictifs
        carbon_stocks = {
            'soil_carbon_stock': np.random.rand(*self.sample_size) * 10,
            'biomass_carbon_stock': np.random.rand(*self.sample_size) * 5,
            'total_carbon_stock': np.random.rand(*self.sample_size) * 15
        }
        carbon_stocks['total_carbon_stock'] = (
            carbon_stocks['soil_carbon_stock'] + carbon_stocks['biomass_carbon_stock']
        )
        
        vulnerability = self.analyzer.assess_carbon_vulnerability(
            carbon_stocks,
            sea_level_rise=0.3,
            temperature_increase=2.0,
            storm_frequency=1.2
        )
        
        required_keys = [
            'erosion_risk', 'thermal_stress', 'storm_disturbance',
            'combined_vulnerability', 'total_potential_loss'
        ]
        for key in required_keys:
            assert key in vulnerability
        
        # Vérifier les plages de valeurs
        assert 0 <= vulnerability['erosion_risk'] <= 1
        assert 0 <= vulnerability['thermal_stress'] <= 1
        assert 0 <= vulnerability['combined_vulnerability'] <= 1


class TestHydrodynamicModel:
    """Tests pour HydrodynamicModel."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.model = HydrodynamicModel(
            location_lat=1.3521,
            location_lon=103.8198,
            pixel_size_m=10.0
        )
    
    def test_initialization(self):
        """Test de l'initialisation du modèle."""
        assert self.model.location_lat == 1.3521
        assert self.model.location_lon == 103.8198
        assert self.model.pixel_size_m == 10.0
        assert self.model.area_per_pixel_ha == 0.01
        
        # Vérifier que les modules sont initialisés
        assert hasattr(self.model, 'time_series_processor')
        assert hasattr(self.model, 'tidal_analyzer')
        assert hasattr(self.model, 'mangrove_classifier')
        assert hasattr(self.model, 'carbon_analyzer')
    
    def test_create_distance_to_coast_map(self):
        """Test de création de carte de distance à la côte."""
        shape = (20, 20)
        distance_map = self.model._create_distance_to_coast_map(shape)
        
        assert distance_map.shape == shape
        assert np.all(distance_map >= 0)
        assert distance_map.dtype == np.float32
    
    def test_create_salinity_proxy(self):
        """Test de création de proxy de salinité."""
        shape = (15, 15)
        inundation_freq = np.random.rand(*shape)
        distance_coast = np.random.rand(*shape) * 1000
        
        salinity_proxy = self.model._create_salinity_proxy(inundation_freq, distance_coast)
        
        assert salinity_proxy.shape == shape
        assert np.all(salinity_proxy >= 0) and np.all(salinity_proxy <= 1)
        assert salinity_proxy.dtype == np.float32
    
    def test_create_elevation_map(self):
        """Test de création de carte d'élévation."""
        shape = (10, 10)
        distance_coast = np.random.rand(*shape) * 1000
        
        elevation_map = self.model._create_elevation_map(shape, distance_coast)
        
        assert elevation_map.shape == shape
        assert np.all(elevation_map >= 0)
        assert elevation_map.dtype == np.float32


def test_module_imports():
    """Test que tous les modules peuvent être importés."""
    try:
        from carbon_dynamics import (
            HydrodynamicModel,
            TimeSeriesProcessor,
            TidalAnalyzer,
            MangroveTypeClassifier,
            CarbonSequestrationAnalyzer
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Erreur d'import: {e}")


def test_integration_synthetic_data():
    """Test d'intégration avec des données synthétiques."""
    # Créer des données synthétiques minimales
    model = HydrodynamicModel(1.0, 100.0, 10.0)
    
    # Données minimales
    shape = (5, 5)
    inundation_freq = np.random.rand(*shape)
    tidal_range = np.full(shape, 2.0)
    distance_coast = np.random.rand(*shape) * 500
    salinity_proxy = np.random.rand(*shape)
    elevation = np.random.rand(*shape) * 2
    
    # Test de création des cartes hydrodynamiques
    hydrodynamic_maps = {
        'inundation_frequency': inundation_freq,
        'tidal_range': tidal_range,
        'distance_to_coast': distance_coast,
        'salinity_proxy': salinity_proxy,
        'elevation': elevation
    }
    
    # Test de classification
    classification = model.classify_mangrove_types(hydrodynamic_maps)
    assert 'prediction_map' in classification
    assert classification['prediction_map'].shape == shape
    
    # Test d'analyse du carbone
    carbon_analysis = model.analyze_carbon_dynamics(classification, hydrodynamic_maps)
    assert 'carbon_report' in carbon_analysis
    assert 'adjusted_sequestration' in carbon_analysis


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
