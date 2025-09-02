"""
Démonstration du module carbon_dynamics avec les vraies données du projet.

Ce script utilise les images générées par le pipeline principal pour analyser
les dynamiques de carbone des mangroves.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_dynamics.hydrodynamic_model import HydrodynamicModel
from config.settings import settings
from core.context import get_engine
from sqlalchemy import text

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_mangrove_locations_from_db(limit=5):
    """
    Récupère des emplacements de mangroves depuis la base de données.
    
    Args:
        limit: Nombre maximum d'emplacements à récupérer
        
    Returns:
        Liste de tuples (lat, lon, description)
    """
    try:
        cfg = settings()
        full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
        
        sql = text(f"""
            SELECT 
                ST_Y(ST_Centroid(geom)) as lat,
                ST_X(ST_Centroid(geom)) as lon,
                ST_Area(geom::geography) / 10000 as area_ha
            FROM {full_table}
            WHERE geom IS NOT NULL 
            AND NOT ST_IsEmpty(geom)
            AND ST_Area(geom::geography) > 1000  -- Au moins 0.1 ha
            ORDER BY ST_Area(geom::geography) DESC
            LIMIT :limit
        """)
        
        locations = []
        with get_engine().connect() as conn:
            for row in conn.execute(sql, {"limit": limit}):
                lat, lon, area_ha = row
                description = f"Zone de {area_ha:.1f} ha"
                locations.append((lat, lon, description))
        
        return locations
        
    except Exception as e:
        logger.warning(f"Impossible de récupérer les emplacements depuis la DB: {e}")
        # Emplacements par défaut (zones connues de mangroves)
        return [
            (1.3521, 103.8198, "Singapour - Sungei Buloh"),
            (1.4404, 103.7924, "Singapour - Lim Chu Kang"),
            (1.2966, 103.7764, "Singapour - Pulau Tekong"),
            (1.3159, 103.9012, "Singapour - Changi"),
            (1.2644, 103.8195, "Singapour - Sisters' Islands")
        ]


def get_generated_images():
    """
    Récupère les images générées par le pipeline principal.
    
    Returns:
        Tuple (image_paths, timestamps, metadata)
    """
    cfg = settings()
    
    # Chercher dans le répertoire de sortie principal
    output_dir = Path(cfg.OUTPUT_DIR)
    image_files = []
    
    # Chercher les images PNG
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        image_files.extend(png_files)
    
    # Chercher aussi dans le répertoire temp pour les TIFF
    temp_dir = output_dir.parent / "data" / "temp"
    if temp_dir.exists():
        for subdir in temp_dir.iterdir():
            if subdir.is_dir():
                tiff_files = list(subdir.glob("*.tif"))
                if tiff_files:
                    # Prendre le premier TIFF de chaque sous-répertoire
                    image_files.append(tiff_files[0])
    
    if not image_files:
        logger.warning("Aucune image trouvée dans les répertoires de sortie")
        return [], [], []
    
    # Limiter à 10 images maximum pour la performance
    image_files = sorted(image_files)[:10]
    image_paths = [str(f) for f in image_files]
    
    # Générer des timestamps basés sur les noms de fichiers ou espacés uniformément
    base_date = datetime(2023, 1, 1)
    timestamps = []
    
    for i, path in enumerate(image_paths):
        # Essayer d'extraire une date du nom de fichier
        filename = Path(path).stem
        
        # Si le nom contient un numéro, l'utiliser pour espacer les dates
        try:
            if "gmw_" in filename:
                number = int(filename.split("_")[1])
                timestamp = base_date + timedelta(days=number * 15)  # Tous les 15 jours
            else:
                timestamp = base_date + timedelta(days=i * 15)
        except:
            timestamp = base_date + timedelta(days=i * 15)
        
        timestamps.append(timestamp)
    
    # Métadonnées basiques
    metadata = []
    for path, timestamp in zip(image_paths, timestamps):
        meta = {
            'path': path,
            'timestamp': timestamp,
            'filename': Path(path).name,
            'size_mb': Path(path).stat().st_size / (1024*1024) if Path(path).exists() else 0
        }
        metadata.append(meta)
    
    return image_paths, timestamps, metadata


def analyze_location(location_data, image_paths, timestamps, output_base_dir):
    """
    Analyse une localisation spécifique.
    
    Args:
        location_data: Tuple (lat, lon, description)
        image_paths: Chemins vers les images
        timestamps: Timestamps des images
        output_base_dir: Répertoire de base pour les sorties
        
    Returns:
        Résultats de l'analyse
    """
    lat, lon, description = location_data
    
    logger.info(f"\n=== ANALYSE DE LA LOCALISATION ===")
    logger.info(f"Coordonnées: {lat:.4f}°N, {lon:.4f}°E")
    logger.info(f"Description: {description}")
    
    # Initialiser le modèle hydrodynamique
    model = HydrodynamicModel(
        location_lat=lat,
        location_lon=lon,
        pixel_size_m=10.0  # Résolution Sentinel-2
    )
    
    # Définir la période d'analyse
    start_date = min(timestamps) if timestamps else datetime(2023, 1, 1)
    end_date = max(timestamps) if timestamps else datetime(2023, 12, 31)
    
    # Scénarios climatiques basés sur les projections IPCC pour l'Asie du Sud-Est
    climate_scenarios = {
        'sea_level_rise': 0.43,        # 43 cm d'ici 2100 (scénario RCP4.5)
        'temperature_increase': 2.3,    # 2.3°C d'augmentation (scénario RCP4.5)
        'storm_frequency': 1.15         # 15% d'augmentation de fréquence des typhons
    }
    
    try:
        # Exécuter l'analyse complète
        results = model.run_complete_analysis(
            image_paths=image_paths,
            timestamps=timestamps,
            start_date=start_date,
            end_date=end_date,
            climate_scenarios=climate_scenarios
        )
        
        # Créer un nom de fichier sûr pour la localisation
        safe_name = f"loc_{lat:.3f}_{lon:.3f}".replace(".", "_").replace("-", "neg")
        
        # Exporter les résultats
        location_output_dir = os.path.join(output_base_dir, safe_name)
        model.export_results(location_output_dir, f"carbon_analysis_{safe_name}")
        
        logger.info(f"Analyse terminée pour {description}")
        logger.info(f"Résultats exportés vers: {location_output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de {description}: {e}")
        return None


def generate_comparative_report(all_results, output_dir):
    """
    Génère un rapport comparatif entre toutes les localisations analysées.
    
    Args:
        all_results: Liste des résultats pour chaque localisation
        output_dir: Répertoire de sortie
    """
    logger.info("\n=== GÉNÉRATION DU RAPPORT COMPARATIF ===")
    
    comparative_data = {
        'locations': [],
        'carbon_summary': [],
        'vulnerability_summary': [],
        'mangrove_types_summary': []
    }
    
    for i, (location_data, results) in enumerate(all_results):
        if results is None:
            continue
            
        lat, lon, description = location_data
        
        # Informations de localisation
        location_info = {
            'id': i + 1,
            'latitude': lat,
            'longitude': lon,
            'description': description
        }
        comparative_data['locations'].append(location_info)
        
        # Résumé carbone
        if 'carbon_analysis' in results:
            carbon_stats = results['carbon_analysis']['carbon_report']['global_statistics']
            carbon_summary = {
                'location_id': i + 1,
                'total_area_ha': carbon_stats.get('total_area_ha', 0),
                'annual_sequestration_MgC': carbon_stats.get('total_annual_sequestration_MgC', 0),
                'total_stock_MgC': carbon_stats.get('total_carbon_stock_MgC', 0),
                'sequestration_rate_MgC_ha_yr': carbon_stats.get('mean_sequestration_rate_MgC_ha_yr', 0),
                'stock_density_MgC_ha': carbon_stats.get('mean_stock_density_MgC_ha', 0)
            }
            comparative_data['carbon_summary'].append(carbon_summary)
            
            # Résumé vulnérabilité
            vulnerability_stats = results['carbon_analysis']['carbon_report']['vulnerability_assessment']
            vulnerability_summary = {
                'location_id': i + 1,
                'mean_vulnerability': vulnerability_stats.get('mean_vulnerability', 0),
                'high_vulnerability_area_ha': vulnerability_stats.get('high_vulnerability_area_ha', 0),
                'potential_carbon_loss_MgC': vulnerability_stats.get('potential_carbon_loss_MgC', 0),
                'carbon_loss_percentage': vulnerability_stats.get('carbon_loss_percentage', 0)
            }
            comparative_data['vulnerability_summary'].append(vulnerability_summary)
        
        # Résumé types de mangroves
        if 'mangrove_classification' in results:
            type_stats = results['mangrove_classification']['type_statistics']
            type_summary = {
                'location_id': i + 1,
                'marine_percentage': type_stats.get('marine', {}).get('percentage', 0),
                'estuarine_percentage': type_stats.get('estuarine', {}).get('percentage', 0),
                'terrestrial_percentage': type_stats.get('terrestrial', {}).get('percentage', 0)
            }
            comparative_data['mangrove_types_summary'].append(type_summary)
    
    # Exporter le rapport comparatif
    import json
    
    report_path = os.path.join(output_dir, "comparative_carbon_analysis.json")
    with open(report_path, 'w') as f:
        json.dump(comparative_data, f, indent=2)
    
    logger.info(f"Rapport comparatif exporté: {report_path}")
    
    # Générer un résumé textuel
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== RÉSUMÉ DE L'ANALYSE DES DYNAMIQUES DE CARBONE ===\n\n")
        
        f.write(f"Nombre de localisations analysées: {len(comparative_data['locations'])}\n\n")
        
        if comparative_data['carbon_summary']:
            f.write("=== STATISTIQUES CARBONE ===\n")
            
            total_area = sum(loc['total_area_ha'] for loc in comparative_data['carbon_summary'])
            total_sequestration = sum(loc['annual_sequestration_MgC'] for loc in comparative_data['carbon_summary'])
            total_stock = sum(loc['total_stock_MgC'] for loc in comparative_data['carbon_summary'])
            
            f.write(f"Surface totale analysée: {total_area:.1f} ha\n")
            f.write(f"Séquestration annuelle totale: {total_sequestration:.1f} Mg C/an\n")
            f.write(f"Stock total de carbone: {total_stock:.1f} Mg C\n")
            
            if total_area > 0:
                avg_sequestration_rate = total_sequestration / total_area
                avg_stock_density = total_stock / total_area
                f.write(f"Taux moyen de séquestration: {avg_sequestration_rate:.2f} Mg C/ha/an\n")
                f.write(f"Densité moyenne de stock: {avg_stock_density:.1f} Mg C/ha\n")
            
            f.write("\n")
        
        if comparative_data['vulnerability_summary']:
            f.write("=== ÉVALUATION DE VULNÉRABILITÉ ===\n")
            
            vulnerabilities = [loc['mean_vulnerability'] for loc in comparative_data['vulnerability_summary']]
            avg_vulnerability = np.mean(vulnerabilities) if vulnerabilities else 0
            
            total_vulnerable_area = sum(loc['high_vulnerability_area_ha'] for loc in comparative_data['vulnerability_summary'])
            total_potential_loss = sum(loc['potential_carbon_loss_MgC'] for loc in comparative_data['vulnerability_summary'])
            
            f.write(f"Vulnérabilité moyenne: {avg_vulnerability:.3f}\n")
            f.write(f"Surface hautement vulnérable: {total_vulnerable_area:.1f} ha\n")
            f.write(f"Perte potentielle de carbone: {total_potential_loss:.1f} Mg C\n")
            
            if vulnerabilities:
                if avg_vulnerability > 0.7:
                    risk_level = "ÉLEVÉ"
                elif avg_vulnerability > 0.4:
                    risk_level = "MODÉRÉ"
                else:
                    risk_level = "FAIBLE"
                f.write(f"Niveau de risque global: {risk_level}\n")
            
            f.write("\n")
        
        if comparative_data['mangrove_types_summary']:
            f.write("=== RÉPARTITION DES TYPES DE MANGROVES ===\n")
            
            type_data = comparative_data['mangrove_types_summary']
            avg_marine = np.mean([loc['marine_percentage'] for loc in type_data])
            avg_estuarine = np.mean([loc['estuarine_percentage'] for loc in type_data])
            avg_terrestrial = np.mean([loc['terrestrial_percentage'] for loc in type_data])
            
            f.write(f"Marine (moyenne): {avg_marine:.1f}%\n")
            f.write(f"Estuarienne (moyenne): {avg_estuarine:.1f}%\n")
            f.write(f"Terrestre (moyenne): {avg_terrestrial:.1f}%\n")
            
            f.write("\n")
        
        f.write("=== RECOMMANDATIONS ===\n")
        f.write("1. Surveillance continue des zones à haute vulnérabilité\n")
        f.write("2. Protection renforcée des zones estuariennes (fort potentiel de carbone)\n")
        f.write("3. Mesures d'adaptation au changement climatique\n")
        f.write("4. Conservation des stocks de carbone existants\n")
        f.write("5. Restauration des zones dégradées\n")
    
    logger.info(f"Résumé textuel exporté: {summary_path}")


def main():
    """
    Fonction principale pour la démonstration complète.
    """
    logger.info("=== DÉMONSTRATION DU MODULE CARBON_DYNAMICS ===")
    logger.info("Utilisation des vraies données du projet")
    
    try:
        # Récupérer les données
        logger.info("\n1. Récupération des données...")
        
        locations = get_mangrove_locations_from_db(limit=3)  # Limiter à 3 pour la démo
        logger.info(f"Localisations trouvées: {len(locations)}")
        
        image_paths, timestamps, metadata = get_generated_images()
        logger.info(f"Images trouvées: {len(image_paths)}")
        
        if not image_paths:
            logger.warning("Aucune image trouvée - la démonstration utilisera des données synthétiques")
            # On pourrait générer des données synthétiques ici si nécessaire
        
        # Préparer le répertoire de sortie
        cfg = settings()
        output_base_dir = os.path.join(cfg.OUTPUT_DIR, "carbon_dynamics_demo")
        os.makedirs(output_base_dir, exist_ok=True)
        
        logger.info(f"Répertoire de sortie: {output_base_dir}")
        
        # Analyser chaque localisation
        logger.info("\n2. Analyse des localisations...")
        
        all_results = []
        
        for i, location in enumerate(locations):
            logger.info(f"\nAnalyse {i+1}/{len(locations)}")
            
            try:
                results = analyze_location(location, image_paths, timestamps, output_base_dir)
                all_results.append((location, results))
                
                # Afficher un résumé rapide
                if results and 'carbon_analysis' in results:
                    carbon_stats = results['carbon_analysis']['carbon_report']['global_statistics']
                    logger.info(f"  → Séquestration: {carbon_stats.get('total_annual_sequestration_MgC', 0):.1f} Mg C/an")
                    logger.info(f"  → Stock total: {carbon_stats.get('total_carbon_stock_MgC', 0):.1f} Mg C")
                
            except Exception as e:
                logger.error(f"Erreur pour la localisation {location}: {e}")
                all_results.append((location, None))
        
        # Générer le rapport comparatif
        logger.info("\n3. Génération du rapport comparatif...")
        generate_comparative_report(all_results, output_base_dir)
        
        # Résumé final
        logger.info("\n=== DÉMONSTRATION TERMINÉE ===")
        successful_analyses = sum(1 for _, results in all_results if results is not None)
        logger.info(f"Analyses réussies: {successful_analyses}/{len(locations)}")
        logger.info(f"Résultats disponibles dans: {output_base_dir}")
        
        logger.info("\nFichiers générés:")
        logger.info("- Analyses individuelles par localisation")
        logger.info("- Rapport comparatif (JSON)")
        logger.info("- Résumé textuel")
        logger.info("- Cartes et visualisations")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {e}")
        raise


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Démonstration réussie!")
    except Exception as e:
        logger.error(f"Échec de la démonstration: {e}")
        sys.exit(1)
