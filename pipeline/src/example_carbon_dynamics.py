"""
Exemple d'utilisation du module carbon_dynamics.

Ce script montre comment utiliser les différents composants du module
pour analyser les dynamiques de carbone des mangroves.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_dynamics import (
    HydrodynamicModel,
    TimeSeriesProcessor,
    TidalAnalyzer,
    MangroveTypeClassifier,
    CarbonSequestrationAnalyzer
)


def create_synthetic_data():
    """
    Crée des données synthétiques pour la démonstration.
    
    Returns:
        Dictionnaire avec les données synthétiques
    """
    print("Création de données synthétiques...")
    
    # Paramètres de la zone d'étude
    height, width = 100, 100  # Taille de la carte en pixels
    pixel_size_m = 10.0
    
    # Créer une carte de fréquence d'inondation réaliste
    # Plus élevée près de la côte (bas de l'image)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Gradient de la côte (en bas) vers l'intérieur (en haut)
    distance_gradient = y_coords / height
    
    # Ajouter de la variabilité spatiale
    noise = np.random.normal(0, 0.1, (height, width))
    inundation_freq = np.clip(
        (1 - distance_gradient) * 0.8 + noise, 
        0, 1
    ).astype(np.float32)
    
    # Créer les autres cartes
    tidal_range = np.full((height, width), 2.5, dtype=np.float32)  # 2.5m d'amplitude
    distance_to_coast = distance_gradient * 1000  # Distance en mètres
    salinity_proxy = inundation_freq * 0.8 + np.random.normal(0, 0.05, (height, width))
    salinity_proxy = np.clip(salinity_proxy, 0, 1).astype(np.float32)
    elevation = distance_gradient * 3 + np.random.normal(0, 0.3, (height, width))
    elevation = np.clip(elevation, 0, 5).astype(np.float32)
    
    # Créer des données de série temporelle
    n_images = 8
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(days=i*45) for i in range(n_images)]
    
    # Simuler des images (3 bandes : Blue, Green, Red)
    images = []
    for i in range(n_images):
        # Simuler une variation saisonnière dans la végétation
        seasonal_factor = 0.8 + 0.2 * np.sin(2 * np.pi * i / n_images)
        
        # Valeurs réflectance typiques (0-1)
        blue = np.random.uniform(0.05, 0.15, (height, width)) * seasonal_factor
        green = np.random.uniform(0.08, 0.20, (height, width)) * seasonal_factor
        red = np.random.uniform(0.06, 0.18, (height, width)) * seasonal_factor
        
        # Simuler l'effet de la végétation (NIR élevé pour végétation dense)
        vegetation_mask = inundation_freq < 0.6  # Végétation là où inondation faible
        nir = np.where(vegetation_mask, 
                      np.random.uniform(0.3, 0.6, (height, width)), 
                      np.random.uniform(0.1, 0.25, (height, width))) * seasonal_factor
        
        # Combiner en image 4 bandes
        image = np.stack([blue, green, red, nir], axis=0)
        images.append(image)
    
    return {
        'images': images,
        'timestamps': timestamps,
        'inundation_frequency': inundation_freq,
        'tidal_range': tidal_range,
        'distance_to_coast': distance_to_coast,
        'salinity_proxy': salinity_proxy,
        'elevation': elevation,
        'pixel_size_m': pixel_size_m,
        'location': {'lat': 1.3521, 'lon': 103.8198}  # Singapour
    }


def example_time_series_analysis():
    """
    Exemple d'analyse de série temporelle.
    """
    print("\n=== EXEMPLE : ANALYSE DE SÉRIE TEMPORELLE ===")
    
    # Créer des données synthétiques
    data = create_synthetic_data()
    
    # Initialiser le processeur
    processor = TimeSeriesProcessor()
    
    # Créer des données de série temporelle
    time_series_data = {
        'timestamps': data['timestamps'],
        'images': data['images'],
        'metadata': [{'timestamp': ts} for ts in data['timestamps']]
    }
    
    # Calculer la fréquence d'inondation
    inundation_freq = processor.calculate_inundation_frequency(time_series_data)
    print(f"Fréquence d'inondation moyenne: {np.mean(inundation_freq):.3f}")
    
    # Analyser les patterns temporels
    patterns = processor.analyze_temporal_patterns(time_series_data)
    print("Patterns temporels calculés")
    
    return inundation_freq, patterns


def example_tidal_analysis():
    """
    Exemple d'analyse des marées.
    """
    print("\n=== EXEMPLE : ANALYSE DES MARÉES ===")
    
    # Initialiser l'analyseur
    data = create_synthetic_data()
    analyzer = TidalAnalyzer(data['location']['lat'], data['location']['lon'])
    
    # Générer des données de marée
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    tidal_data = analyzer.get_tidal_data_theoretical(start_date, end_date)
    print(f"Données de marée générées: {len(tidal_data)} points")
    
    # Classifier les conditions
    classified = analyzer.classify_tidal_conditions(tidal_data)
    
    # Statistiques des conditions
    condition_counts = classified['tidal_condition'].value_counts()
    print("Répartition des conditions de marée:")
    for condition, count in condition_counts.items():
        percentage = (count / len(classified)) * 100
        print(f"  - {condition}: {percentage:.1f}%")
    
    # Calculer l'amplitude
    with_range = analyzer.calculate_tidal_range(classified)
    mean_range = with_range['tidal_range'].mean()
    print(f"Amplitude moyenne des marées: {mean_range:.2f} m")
    
    # Analyser les patterns saisonniers
    seasonal = analyzer.analyze_seasonal_patterns(with_range)
    print("Patterns saisonniers calculés")
    
    return with_range, seasonal


def example_mangrove_classification():
    """
    Exemple de classification des mangroves.
    """
    print("\n=== EXEMPLE : CLASSIFICATION DES MANGROVES ===")
    
    # Créer des données synthétiques
    data = create_synthetic_data()
    
    # Initialiser le classificateur
    classifier = MangroveTypeClassifier()
    
    # Classifier la carte
    results = classifier.classify_mangrove_map(
        inundation_frequency=data['inundation_frequency'],
        tidal_range=data['tidal_range'],
        distance_to_coast=data['distance_to_coast'],
        salinity_proxy=data['salinity_proxy'],
        elevation=data['elevation']
    )
    
    print("Classification des mangroves:")
    for mangrove_type, stats in results['type_statistics'].items():
        print(f"  - {mangrove_type}: {stats['percentage']:.1f}% ({stats['count']} pixels)")
    
    return results


def example_carbon_analysis():
    """
    Exemple d'analyse du carbone.
    """
    print("\n=== EXEMPLE : ANALYSE DU CARBONE ===")
    
    # Créer des données synthétiques
    data = create_synthetic_data()
    
    # Classifier les mangroves d'abord
    classifier = MangroveTypeClassifier()
    classification = classifier.classify_mangrove_map(
        inundation_frequency=data['inundation_frequency'],
        tidal_range=data['tidal_range'],
        distance_to_coast=data['distance_to_coast'],
        salinity_proxy=data['salinity_proxy'],
        elevation=data['elevation']
    )
    
    # Initialiser l'analyseur de carbone
    carbon_analyzer = CarbonSequestrationAnalyzer()
    
    # Calculer la surface par pixel
    pixel_size_m = data['pixel_size_m']
    area_per_pixel_ha = (pixel_size_m * pixel_size_m) / 10000
    
    # Calculer la séquestration de base
    base_sequestration = carbon_analyzer.calculate_base_sequestration(
        classification['prediction_map'], area_per_pixel_ha
    )
    
    print(f"Séquestration de base moyenne: {np.mean(base_sequestration):.3f} Mg C/pixel/an")
    
    # Appliquer les ajustements environnementaux
    adjusted_sequestration = carbon_analyzer.apply_environmental_adjustments(
        base_sequestration,
        salinity_map=data['salinity_proxy'] * 35,  # Conversion en ppt
        tidal_range_map=data['tidal_range'],
        inundation_freq_map=data['inundation_frequency']
    )
    
    print(f"Séquestration ajustée moyenne: {np.mean(adjusted_sequestration):.3f} Mg C/pixel/an")
    
    # Calculer les stocks
    carbon_stocks = carbon_analyzer.calculate_carbon_stocks(adjusted_sequestration)
    
    print(f"Stock total de carbone: {np.sum(carbon_stocks['total_carbon_stock']):.1f} Mg C")
    print(f"Répartition - Sol: {carbon_stocks['soil_percentage']:.1f}%, Biomasse: {carbon_stocks['biomass_percentage']:.1f}%")
    
    # Évaluer la vulnérabilité
    vulnerability = carbon_analyzer.assess_carbon_vulnerability(
        carbon_stocks,
        sea_level_rise=0.5,
        temperature_increase=2.0,
        storm_frequency=1.2
    )
    
    print(f"Vulnérabilité moyenne: {np.mean(vulnerability['combined_vulnerability']):.3f}")
    print(f"Perte potentielle: {vulnerability['loss_percentage']:.1f}% du stock")
    
    return {
        'sequestration': adjusted_sequestration,
        'stocks': carbon_stocks,
        'vulnerability': vulnerability
    }


def example_complete_hydrodynamic_model():
    """
    Exemple d'utilisation du modèle hydrodynamique complet.
    """
    print("\n=== EXEMPLE : MODÈLE HYDRODYNAMIQUE COMPLET ===")
    
    # Créer des données synthétiques
    data = create_synthetic_data()
    
    # Initialiser le modèle
    model = HydrodynamicModel(
        location_lat=data['location']['lat'],
        location_lon=data['location']['lon'],
        pixel_size_m=data['pixel_size_m']
    )
    
    # Simuler des chemins d'images (pour la démonstration)
    image_paths = [f"synthetic_image_{i}.tif" for i in range(len(data['timestamps']))]
    
    # Remplacer la méthode de chargement pour utiliser nos données synthétiques
    def mock_load_time_series(paths, timestamps):
        return {
            'timestamps': data['timestamps'],
            'images': data['images'],
            'metadata': [{'timestamp': ts} for ts in data['timestamps']]
        }
    
    model.time_series_processor.load_time_series = mock_load_time_series
    
    # Exécuter l'analyse complète
    results = model.run_complete_analysis(
        image_paths=image_paths,
        timestamps=data['timestamps'],
        start_date=data['timestamps'][0],
        end_date=data['timestamps'][-1],
        climate_scenarios={
            'sea_level_rise': 0.5,
            'temperature_increase': 2.5,
            'storm_frequency': 1.3
        }
    )
    
    # Afficher les résultats
    print("RÉSULTATS DE L'ANALYSE COMPLÈTE:")
    
    # Classification
    if 'mangrove_classification' in results:
        type_stats = results['mangrove_classification']['type_statistics']
        print("\nRépartition des types de mangroves:")
        for mangrove_type, stats in type_stats.items():
            print(f"  - {mangrove_type}: {stats['percentage']:.1f}%")
    
    # Carbone
    if 'carbon_analysis' in results:
        carbon_stats = results['carbon_analysis']['carbon_report']['global_statistics']
        print(f"\nAnalyse du carbone:")
        print(f"  - Surface totale: {carbon_stats['total_area_ha']:.1f} ha")
        print(f"  - Séquestration annuelle: {carbon_stats['total_annual_sequestration_MgC']:.1f} Mg C/an")
        print(f"  - Stock total: {carbon_stats['total_carbon_stock_MgC']:.1f} Mg C")
        print(f"  - Taux moyen: {carbon_stats['mean_sequestration_rate_MgC_ha_yr']:.2f} Mg C/ha/an")
    
    return results


def create_visualizations(results):
    """
    Crée des visualisations des résultats.
    
    Args:
        results: Résultats de l'analyse complète
    """
    print("\n=== CRÉATION DES VISUALISATIONS ===")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Analyse des Dynamiques de Carbone des Mangroves', fontsize=16)
        
        # 1. Fréquence d'inondation
        if 'hydrodynamic_maps' in results:
            maps = results['hydrodynamic_maps']
            
            im1 = axes[0, 0].imshow(maps['inundation_frequency'], cmap='Blues')
            axes[0, 0].set_title('Fréquence d\'inondation')
            axes[0, 0].set_xlabel('Pixels')
            axes[0, 0].set_ylabel('Pixels')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            
            # 2. Distance à la côte
            im2 = axes[0, 1].imshow(maps['distance_to_coast'], cmap='viridis')
            axes[0, 1].set_title('Distance à la côte (m)')
            axes[0, 1].set_xlabel('Pixels')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
            
            # 3. Proxy de salinité
            im3 = axes[0, 2].imshow(maps['salinity_proxy'], cmap='plasma')
            axes[0, 2].set_title('Proxy de salinité')
            axes[0, 2].set_xlabel('Pixels')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # 4. Classification des mangroves
        if 'mangrove_classification' in results:
            classification = results['mangrove_classification']
            prediction_map = classification['prediction_map']
            
            # Créer une carte colorée pour les types
            colors = np.array([[0.2, 0.6, 0.8],    # Marine: bleu
                              [0.2, 0.8, 0.2],     # Estuarine: vert
                              [0.8, 0.6, 0.2]])    # Terrestrial: brun
            
            colored_map = colors[prediction_map]
            
            axes[1, 0].imshow(colored_map)
            axes[1, 0].set_title('Types de mangroves')
            axes[1, 0].set_xlabel('Pixels')
            axes[1, 0].set_ylabel('Pixels')
            
            # Légende
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[0], label='Marine'),
                             Patch(facecolor=colors[1], label='Estuarienne'),
                             Patch(facecolor=colors[2], label='Terrestre')]
            axes[1, 0].legend(handles=legend_elements, loc='upper right')
        
        # 5. Séquestration de carbone
        if 'carbon_analysis' in results:
            carbon = results['carbon_analysis']
            
            im5 = axes[1, 1].imshow(carbon['adjusted_sequestration'], cmap='Greens')
            axes[1, 1].set_title('Séquestration de carbone\n(Mg C/pixel/an)')
            axes[1, 1].set_xlabel('Pixels')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
            
            # 6. Vulnérabilité
            vulnerability_map = carbon['vulnerability_assessment']['combined_vulnerability']
            im6 = axes[1, 2].imshow(vulnerability_map, cmap='Reds')
            axes[1, 2].set_title('Vulnérabilité climatique')
            axes[1, 2].set_xlabel('Pixels')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = "carbon_dynamics_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {output_path}")
        
        # Afficher si possible
        try:
            plt.show()
        except:
            print("Impossible d'afficher les graphiques (environnement sans GUI)")
        
    except ImportError:
        print("Matplotlib non disponible pour les visualisations")
    except Exception as e:
        print(f"Erreur lors de la création des visualisations: {e}")


def main():
    """
    Fonction principale pour exécuter tous les exemples.
    """
    print("=== EXEMPLES D'UTILISATION DU MODULE CARBON_DYNAMICS ===")
    
    try:
        # Exemples individuels
        inundation_freq, patterns = example_time_series_analysis()
        tidal_data, seasonal = example_tidal_analysis()
        classification = example_mangrove_classification()
        carbon_results = example_carbon_analysis()
        
        # Exemple complet
        complete_results = example_complete_hydrodynamic_model()
        
        # Créer des visualisations
        create_visualizations(complete_results)
        
        print("\n=== TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS ===")
        print("Le module carbon_dynamics est prêt à être utilisé!")
        
        return complete_results
        
    except Exception as e:
        print(f"Erreur lors de l'exécution des exemples: {e}")
        raise


if __name__ == "__main__":
    results = main()
