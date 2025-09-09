"""
Script principal pour analyser les dynamiques de carbone des mangroves.

Ce script utilise le module carbon_dynamics pour analyser les stocks et flux de carbone
en fonction des conditions hydrodynamiques et des marées.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_dynamics.hydrodynamic_model import HydrodynamicModel
from pipeline.src.config.settings_s2 import settings_s2

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sample_data():
    """
    Récupère des données d'exemple pour tester l'analyse.
    
    Returns:
        Tuple (image_paths, timestamps)
    """
    cfg = settings_s2()
    data_dir = Path(cfg.OUTPUT_DIR_TIME_SERIES)
    
    # Chercher les images générées
    image_paths = []
    image_files = list(data_dir.glob("*.png"))
    
    if not image_files:
        logger.warning("Aucune image trouvée dans le répertoire de sortie")
        # Créer des données d'exemple
        image_paths = [str(data_dir / f"gmw_{i}.png") for i in range(1, 6)]
    else:
        image_paths = [str(f) for f in sorted(image_files)[:10]]  # Maximum 10 images
    
    # Générer des timestamps correspondants (une image tous les 15 jours)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(days=i*15) for i in range(len(image_paths))]
    
    return image_paths, timestamps


def main():
    """
    Fonction principale pour l'analyse des dynamiques de carbone.
    """
    logger.info("=== ANALYSE DES DYNAMIQUES DE CARBONE DES MANGROVES ===")
    
    try:
        # Configuration
        cfg = settings_s2()
        
        # Coordonnées d'exemple (Singapour - zone de mangroves)
        location_lat = 1.3521  # Latitude de Singapour
        location_lon = 103.8198  # Longitude de Singapour
        pixel_size_m = 10.0  # Résolution Sentinel-2
        
        logger.info(f"Zone d'étude: {location_lat}°N, {location_lon}°E")
        logger.info(f"Résolution spatiale: {pixel_size_m}m")
        
        # Initialiser le modèle hydrodynamique
        hydro_model = HydrodynamicModel(
            location_lat=location_lat,
            location_lon=location_lon,
            pixel_size_m=pixel_size_m
        )
        
        # Récupérer les données d'exemple
        image_paths, timestamps = get_sample_data()
        logger.info(f"Nombre d'images à analyser: {len(image_paths)}")
        
        # Vérifier que les fichiers existent
        existing_paths = []
        existing_timestamps = []
        for path, timestamp in zip(image_paths, timestamps):
            if os.path.exists(path):
                existing_paths.append(path)
                existing_timestamps.append(timestamp)
            else:
                logger.warning(f"Fichier non trouvé: {path}")
        
        if not existing_paths:
            logger.error("Aucun fichier image trouvé. Génération de données synthétiques...")
            # Créer des données synthétiques pour la démonstration
            existing_paths = ["synthetic_data"] * 5
            existing_timestamps = [datetime(2023, 1, 1) + timedelta(days=i*15) for i in range(5)]
        
        # Définir la période d'analyse
        start_analysis = min(existing_timestamps)
        end_analysis = max(existing_timestamps)
        
        logger.info(f"Période d'analyse: {start_analysis} à {end_analysis}")
        
        # Scénarios climatiques pour l'évaluation de vulnérabilité
        climate_scenarios = {
            'sea_level_rise': 0.5,        # 50 cm d'élévation du niveau de la mer
            'temperature_increase': 2.5,   # 2.5°C d'augmentation de température
            'storm_frequency': 1.3         # 30% d'augmentation de la fréquence des tempêtes
        }
        
        logger.info("Scénarios climatiques:")
        for scenario, value in climate_scenarios.items():
            logger.info(f"  - {scenario}: {value}")
        
        # Exécuter l'analyse complète
        logger.info("Démarrage de l'analyse hydrodynamique complète...")
        
        results = hydro_model.run_complete_analysis(
            image_paths=existing_paths,
            timestamps=existing_timestamps,
            start_date=start_analysis,
            end_date=end_analysis,
            climate_scenarios=climate_scenarios
        )
        
        # Afficher les résultats principaux
        logger.info("\n=== RÉSULTATS PRINCIPAUX ===")
        
        if 'mangrove_classification' in results:
            type_stats = results['mangrove_classification']['type_statistics']
            logger.info("Répartition des types de mangroves:")
            for mangrove_type, stats in type_stats.items():
                logger.info(f"  - {mangrove_type}: {stats['percentage']:.1f}% ({stats['count']} pixels)")
        
        if 'carbon_analysis' in results:
            carbon_stats = results['carbon_analysis']['carbon_report']['global_statistics']
            logger.info("\nAnalyse du carbone:")
            logger.info(f"  - Surface totale: {carbon_stats['total_area_ha']:.1f} ha")
            logger.info(f"  - Séquestration annuelle: {carbon_stats['total_annual_sequestration_MgC']:.1f} Mg C/an")
            logger.info(f"  - Stock total de carbone: {carbon_stats['total_carbon_stock_MgC']:.1f} Mg C")
            logger.info(f"  - Taux moyen de séquestration: {carbon_stats['mean_sequestration_rate_MgC_ha_yr']:.2f} Mg C/ha/an")
            
            vulnerability_stats = results['carbon_analysis']['carbon_report']['vulnerability_assessment']
            logger.info(f"\nVulnérabilité climatique:")
            logger.info(f"  - Vulnérabilité moyenne: {vulnerability_stats['mean_vulnerability']:.2f}")
            logger.info(f"  - Perte potentielle de carbone: {vulnerability_stats['potential_carbon_loss_MgC']:.1f} Mg C ({vulnerability_stats['carbon_loss_percentage']:.1f}%)")
            logger.info(f"  - Zone hautement vulnérable: {vulnerability_stats['high_vulnerability_area_ha']:.1f} ha")
        
        if 'tidal_correlations' in results:
            correlations = results['tidal_correlations']['spectral_correlations']
            if correlations:
                logger.info("\nCorreptions marées-végétation:")
                for index, corr_data in correlations.items():
                    if isinstance(corr_data, dict) and 'pearson_correlation' in corr_data:
                        corr_value = corr_data['pearson_correlation']
                        logger.info(f"  - {index.upper()}: r = {corr_value:.3f}")
        
        # Exporter les résultats
        output_dir = os.path.join(cfg.OUTPUT_DIR, "carbon_dynamics_analysis")
        hydro_model.export_results(output_dir, "mangrove_carbon_dynamics")
        
        logger.info(f"\n=== ANALYSE TERMINÉE ===")
        logger.info(f"Résultats exportés vers: {output_dir}")
        
        # Recommandations basées sur les résultats
        logger.info("\n=== RECOMMANDATIONS ===")
        
        if 'carbon_analysis' in results:
            vulnerability_stats = results['carbon_analysis']['carbon_report']['vulnerability_assessment']
            vuln_level = vulnerability_stats['mean_vulnerability']
            
            if vuln_level > 0.7:
                logger.info("🔴 VULNÉRABILITÉ ÉLEVÉE:")
                logger.info("  - Mise en place urgente de mesures de protection")
                logger.info("  - Surveillance accrue des zones à risque")
                logger.info("  - Stratégies d'adaptation climatique nécessaires")
            elif vuln_level > 0.4:
                logger.info("🟡 VULNÉRABILITÉ MODÉRÉE:")
                logger.info("  - Surveillance régulière recommandée")
                logger.info("  - Planification préventive des mesures de protection")
            else:
                logger.info("🟢 VULNÉRABILITÉ FAIBLE:")
                logger.info("  - Maintien des pratiques de conservation actuelles")
                logger.info("  - Surveillance de routine suffisante")
            
            # Recommandations par type de mangrove
            type_stats = results['mangrove_classification']['type_statistics']
            if 'estuarine' in type_stats and type_stats['estuarine']['percentage'] > 50:
                logger.info("\n🌊 DOMINANCE ESTUARIENNE:")
                logger.info("  - Fort potentiel de séquestration de carbone")
                logger.info("  - Surveillance de la qualité de l'eau recommandée")
                logger.info("  - Protection contre la pollution urbaine/industrielle")
            
            if 'marine' in type_stats and type_stats['marine']['percentage'] > 30:
                logger.info("\n🌊 EXPOSITION MARINE IMPORTANTE:")
                logger.info("  - Vulnérabilité à l'élévation du niveau de la mer")
                logger.info("  - Mesures de protection côtière à envisager")
                logger.info("  - Surveillance de l'érosion côtière")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {e}")
        raise


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Script terminé avec succès")
    except Exception as e:
        logger.error(f"Échec du script: {e}")
        sys.exit(1)
