#!/usr/bin/env python3
"""
Script de validation des stocks de carbone des mangroves au Vietnam
Compare les estimations de l'article avec les données de la base PostgreSQL
Adapté pour le delta du Fleuve Rouge (Vietnam) basé sur l'article fourni.
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "soc_pipeline"))

from soc_pipeline.config.settings import settings_s1

def connect_to_db():
    """Se connecter à la base de données PostgreSQL"""
    config = settings_s1()
    try:
        conn = psycopg2.connect(
            host=config.PG_HOST,
            port=config.PG_PORT,
            database=config.PG_DB,
            user=config.PG_USER,
            password=config.PG_PASSWORD
        )
        print("Connexion à la base réussie.")
        return conn
    except Exception as e:
        print(f"Erreur de connexion à la base : {e}")
        return None

def explore_table_structure():
    """Explorer la structure de la table mangrove_carbon"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        print("=== Structure de la table mangrove_carbon ===")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'mangrove_carbon' 
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        for col in columns:
            print(f"- {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
        
        cursor.execute("SELECT COUNT(*) FROM mangrove_carbon;")
        total_rows = cursor.fetchone()[0]
        print(f"\nNombre total d'enregistrements (rasters) : {total_rows}")
        
        # Plage géographique globale en utilisant l'étendue des rasters
        cursor.execute("""
            SELECT ST_AsText(ST_Extent(ST_Envelope(rast))) FROM mangrove_carbon;
        """)
        extent_box = cursor.fetchone()[0]
        print(f"\nPlage géographique globale (Bounding Box) : {extent_box}")
        
        # Échantillon de données
        print("\n=== Échantillon de métadonnées des rasters ===")
        cursor.execute("""
            SELECT 
                id, 
                name, 
                ST_Width(rast) as width, 
                ST_Height(rast) as height, 
                ST_SRID(rast) as srid
            FROM mangrove_carbon 
            LIMIT 5;
        """)
        sample_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df_sample = pd.DataFrame(sample_data, columns=column_names)
        print(df_sample.to_string())
        
    except Exception as e:
        print(f"Erreur lors de l'exploration : {e}")
    finally:
        conn.close()

def get_province_data(province_name, lat_min, lat_max, lon_min, lon_max):
    """
    Récupérer les données SOC pour une province spécifique
    
    Args:
        province_name: Nom de la province
        lat_min, lat_max: Limites de latitude
        lon_min, lon_max: Limites de longitude
    """
    conn = connect_to_db()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Requête spatiale pour extraire les valeurs du raster
        query = """
            WITH bbox AS (
                SELECT ST_SetSRID(
                    ST_MakeEnvelope(%s, %s, %s, %s), 4326
                ) AS geom
            ),
            clipped_rasters AS (
                SELECT ST_Clip(t.rast, b.geom) AS clipped_rast
                FROM mangrove_carbon t, bbox b
                WHERE ST_Intersects(t.rast, b.geom)
            )
            SELECT unnest((ST_DumpValues(clipped_rast)).valarray) AS pixel_value
            FROM clipped_rasters;
        """
        
        cursor.execute(query, (lon_min, lat_min, lon_max, lat_max))
        data = cursor.fetchall()
        
        # Les données sont maintenant une liste de tuples avec les valeurs de pixels
        # On extrait les valeurs et on filtre les potentiels None ou valeurs non-valides
        soc_values = [row[0] for row in data if row and row[0] is not None and np.isfinite(row[0])]
        
        if not soc_values:
            print(f"\n=== Aucune donnée SOC trouvée pour {province_name}. ===")
            return None

        df = pd.DataFrame(soc_values, columns=['soc'])
        
        print(f"\n=== Données trouvées pour {province_name} ===")
        print(f"Zone géographique : {lat_min:.2f}°N - {lat_max:.2f}°N, {lon_min:.2f}°E - {lon_max:.2f}°E")
        print(f"Nombre de pixels valides : {len(df)}")
        
        if len(df) > 0:
            print(f"SOC moyen : {df['soc'].mean():.2f} Mg ha⁻¹")
            print(f"SOC min/max : {df['soc'].min():.2f} / {df['soc'].max():.2f} Mg ha⁻¹")
            print(f"SOC médian : {df['soc'].median():.2f} Mg ha⁻¹")
            print(f"Écart-type : {df['soc'].std():.2f} Mg ha⁻¹")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données pour {province_name} : {e}")
        return None
    finally:
        conn.close()

def get_vietnam_data():
    """
    Récupérer les données de la zone du delta du Fleuve Rouge (Vietnam)
    Basé sur l'article : environ 20°00′–20°44′ N et 106°01′–106°39′ E
    """
    conn = connect_to_db()
    if not conn:
        return None
    
    try:
        # Zone approximative du delta du Fleuve Rouge
        lat_min, lat_max = 20.00, 20.73
        lon_min, lon_max = 106.02, 106.65
        
        cursor = conn.cursor()
        
        # Requête spatiale pour extraire les valeurs du raster
        query = """
            WITH bbox AS (
                SELECT ST_SetSRID(
                    ST_MakeEnvelope(%s, %s, %s, %s), 4326
                ) AS geom
            ),
            clipped_rasters AS (
                SELECT ST_Clip(t.rast, b.geom) AS clipped_rast
                FROM mangrove_carbon t, bbox b
                WHERE ST_Intersects(t.rast, b.geom)
            )
            SELECT unnest((ST_DumpValues(clipped_rast)).valarray) AS pixel_value
            FROM clipped_rasters;
        """
        
        cursor.execute(query, (lon_min, lat_min, lon_max, lat_max))
        data = cursor.fetchall()
        
        # Les données sont maintenant une liste de tuples avec les valeurs de pixels
        # On extrait les valeurs et on filtre les potentiels None ou valeurs non-valides
        soc_values = [row[0] for row in data if row and row[0] is not None and np.isfinite(row[0])]
        
        if not soc_values:
            print("\n=== Aucune donnée SOC trouvée pour la zone spécifiée. ===")
            return None

        df = pd.DataFrame(soc_values, columns=['soc'])
        
        print(f"\n=== Données trouvées pour le delta du Fleuve Rouge (Vietnam) - Zone globale ===")
        print(f"Nombre de pixels valides : {len(df)}")
        
        if len(df) > 0:
            print(f"SOC moyen : {df['soc'].mean():.2f} Mg ha⁻¹")
            print(f"SOC min/max : {df['soc'].min():.2f} / {df['soc'].max():.2f} Mg ha⁻¹")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None
    finally:
        conn.close()

def compare_with_article_data(article_soc_stats, province_data):
    """
    Comparaison des données de chaque province avec les valeurs de l'article
    
    Args:
        article_soc_stats: Dict avec stats de l'article pour chaque province
        province_data: Dict avec les DataFrames de chaque province
    """
    
    print("\n=== Comparaison détaillée par province (en Mg ha⁻¹) ===")
    
    for province_name, stats in article_soc_stats.items():
        if province_name in province_data and province_data[province_name] is not None:
            df = province_data[province_name]
            
            # Stats de la base pour cette province
            db_mean = df['soc'].mean()
            db_min = df['soc'].min()
            db_max = df['soc'].max()
            db_median = df['soc'].median()
            db_std = df['soc'].std()
            
            print(f"\n{'='*50}")
            print(f"Province : {province_name}")
            print(f"{'='*50}")
            print(f"Article :")
            print(f"  - Moyenne: {stats['mean']:.2f} Mg ha⁻¹")
            print(f"  - Min: {stats['min']:.2f} Mg ha⁻¹") 
            print(f"  - Max: {stats['max']:.2f} Mg ha⁻¹")
            
            print(f"\nBase de données :")
            print(f"  - Moyenne: {db_mean:.2f} Mg ha⁻¹")
            print(f"  - Médiane: {db_median:.2f} Mg ha⁻¹")
            print(f"  - Min: {db_min:.2f} Mg ha⁻¹")
            print(f"  - Max: {db_max:.2f} Mg ha⁻¹")
            print(f"  - Écart-type: {db_std:.2f} Mg ha⁻¹")
            print(f"  - Nombre de pixels: {len(df):,}")
            
            print(f"\nComparaison :")
            diff_mean = abs(stats['mean'] - db_mean)
            diff_percent = (diff_mean / stats['mean']) * 100
            print(f"  - Différence moyenne: {diff_mean:.2f} Mg ha⁻¹ ({diff_percent:.1f}%)")
            
            # Analyser si la moyenne de l'article est dans la plage de variation de la base
            if stats['mean'] >= db_min and stats['mean'] <= db_max:
                print(f"  ✓ La moyenne de l'article ({stats['mean']:.2f}) est dans la plage de la base")
            else:
                print(f"  ✗ La moyenne de l'article ({stats['mean']:.2f}) est hors de la plage de la base")
                
        else:
            print(f"\n⚠️  Aucune donnée disponible pour {province_name}")
            
    print(f"\n{'='*70}")

def analyze_spatial_distribution():
    """
    Analyser la distribution spatiale des données SOC dans la région
    """
    print("\n=== Analyse spatiale du delta du Fleuve Rouge ===")
    
    # Coordonnées approximatives basées sur la géographie du Vietnam
    # Nam Dinh : plus au sud du delta
    nam_dinh_coords = {
        'lat_min': 20.00, 'lat_max': 20.30,
        'lon_min': 106.02, 'lon_max': 106.35
    }
    
    # Thai Binh : plus au nord du delta  
    thai_binh_coords = {
        'lat_min': 20.30, 'lat_max': 20.73,
        'lon_min': 106.15, 'lon_max': 106.65
    }
    
    # Zone globale (comme référence)
    global_coords = {
        'lat_min': 20.00, 'lat_max': 20.73,
        'lon_min': 106.02, 'lon_max': 106.65
    }
    
    # Récupérer les données pour chaque zone
    province_data = {}
    
    print("\nRécupération des données par province...")
    province_data['Nam Dinh'] = get_province_data(
        'Nam Dinh', 
        nam_dinh_coords['lat_min'], nam_dinh_coords['lat_max'],
        nam_dinh_coords['lon_min'], nam_dinh_coords['lon_max']
    )
    
    province_data['Thai Binh'] = get_province_data(
        'Thai Binh',
        thai_binh_coords['lat_min'], thai_binh_coords['lat_max'], 
        thai_binh_coords['lon_min'], thai_binh_coords['lon_max']
    )
    
    province_data['Global (Delta complet)'] = get_province_data(
        'Global (Delta complet)',
        global_coords['lat_min'], global_coords['lat_max'],
        global_coords['lon_min'], global_coords['lon_max'] 
    )
    
    return province_data

if __name__ == "__main__":
    print("=== Validation des stocks de carbone des mangroves au Vietnam ===\n")
    
    # 1. Explorer la structure de la base
    explore_table_structure()
    
    # 2. Analyser la distribution spatiale par province
    province_data = analyze_spatial_distribution()
    
    # 3. Stats de l'article (extraites de Table 4)
    article_soc_stats = {
        'Nam Dinh': {'mean': 55.42, 'min': 4.84, 'max': 84.04},
        'Thai Binh': {'mean': 90.37, 'min': 44.82, 'max': 158.89},
        'Global (Delta complet)': {'mean': 68.76, 'min': 44.74, 'max': 91.92}
    }
    
    # 4. Comparer avec l'article de manière spatialement cohérente
    compare_with_article_data(article_soc_stats, province_data)
    

