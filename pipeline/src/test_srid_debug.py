#!/usr/bin/env python3
"""
Script de diagnostic pour les problèmes SRID dans PostgreSQL.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from database.gmw_v3 import get_engine

def test_postgis_setup():
    """Test de la configuration PostGIS de base"""
    print("=== Test de la configuration PostGIS ===")
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Vérifier la version de PostGIS
            result = conn.execute(text("SELECT PostGIS_Version();")).scalar()
            print(f"PostGIS Version: {result}")
            
            # Vérifier les SRID disponibles
            result = conn.execute(text("SELECT COUNT(*) FROM spatial_ref_sys WHERE srid IN (4326, 3857);")).scalar()
            print(f"SRID 4326 et 3857 disponibles: {result == 2}")
            
            # Test simple de création de géométrie
            result = conn.execute(text("SELECT ST_AsText(ST_Point(5.5, 5.9));")).scalar()
            print(f"Création de point simple: {result}")
            
    except Exception as e:
        print(f"Erreur lors du test PostGIS: {e}")
        return False
    
    return True

def test_srid_methods():
    """Teste différentes méthodes pour définir le SRID"""
    print("\n=== Test des méthodes SRID ===")
    
    try:
        engine = get_engine()
        
        test_coords = "POLYGON((5.5 5.9, 5.6 5.9, 5.6 6.0, 5.5 6.0, 5.5 5.9))"
        
        methods = [
            ("ST_SetSRID + ST_GeomFromText", f"ST_SetSRID(ST_GeomFromText('{test_coords}'), 4326)"),
            ("ST_GeomFromText avec SRID param", f"ST_GeomFromText('{test_coords}', 4326)"),
            ("ST_GeomFromEWKT", f"ST_GeomFromEWKT('SRID=4326;{test_coords}')"),
        ]
        
        with engine.connect() as conn:
            for method_name, sql_fragment in methods:
                try:
                    # Test de création de géométrie
                    sql = text(f"SELECT ST_SRID({sql_fragment}) as srid, ST_AsText({sql_fragment}) as wkt;")
                    result = conn.execute(sql).fetchone()
                    print(f"✓ {method_name}: SRID={result.srid}, WKT={result.wkt[:50]}...")
                    
                    # Test de transformation
                    sql_transform = text(f"SELECT ST_AsText(ST_Transform({sql_fragment}, 3857)) as transformed;")
                    result_transform = conn.execute(sql_transform).scalar()
                    print(f"  Transformation vers 3857: {result_transform[:50]}...")
                    
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"Erreur lors du test des méthodes SRID: {e}")
        return False
    
    return True

def test_gmw_table():
    """Teste la table GMW directement"""
    print("\n=== Test de la table GMW ===")
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Vérifier l'existence de la table
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'gmw_v3_2020_vec';
            """)).scalar()
            
            if result == 0:
                print("✗ Table gmw_v3_2020_vec introuvable")
                return False
            
            print(f"✓ Table gmw_v3_2020_vec trouvée")
            
            # Vérifier les colonnes géométriques
            result = conn.execute(text("""
                SELECT column_name, udt_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'gmw_v3_2020_vec'
                AND udt_name = 'geometry';
            """)).fetchall()
            
            for row in result:
                print(f"✓ Colonne géométrique: {row.column_name}")
            
            # Vérifier le SRID de la géométrie
            result = conn.execute(text("""
                SELECT ST_SRID(geom) as srid, COUNT(*) as count
                FROM public.gmw_v3_2020_vec
                GROUP BY ST_SRID(geom)
                LIMIT 5;
            """)).fetchall()
            
            for row in result:
                print(f"✓ SRID dans la table: {row.srid} ({row.count} enregistrements)")
            
    except Exception as e:
        print(f"Erreur lors du test de la table GMW: {e}")
        return False
    
    return True

def test_coverage_query():
    """Teste la requête de couverture avec différentes approches"""
    print("\n=== Test de la requête de couverture ===")
    
    try:
        engine = get_engine()
        
        # BBox de test autour de la Côte d'Ivoire (zone connue avec des mangroves)
        test_bbox = "POLYGON((-5.5 5.0, -5.0 5.0, -5.0 5.5, -5.5 5.5, -5.5 5.0))"
        
        methods = [
            ("Méthode 1: ST_GeomFromText + SRID param", 
             f"ST_GeomFromText('{test_bbox}', 4326)"),
            ("Méthode 2: ST_SetSRID + ST_GeomFromText", 
             f"ST_SetSRID(ST_GeomFromText('{test_bbox}'), 4326)"),
        ]
        
        with engine.connect() as conn:
            for method_name, geom_creation in methods:
                try:
                    sql = text(f"""
                        WITH bbox_geom AS (
                            SELECT ST_Transform({geom_creation}, 3857) as geom
                        ),
                        bbox_area AS (
                            SELECT ST_Area(geom) as total_area, geom FROM bbox_geom
                        )
                        SELECT total_area FROM bbox_area;
                    """)
                    
                    result = conn.execute(sql).scalar()
                    print(f"✓ {method_name}: Aire calculée = {result:.2f} m²")
                    
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
        
    except Exception as e:
        print(f"Erreur lors du test de la requête: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Diagnostic des problèmes SRID PostgreSQL/PostGIS\n")
    
    success = True
    success &= test_postgis_setup()
    success &= test_srid_methods() 
    success &= test_gmw_table()
    success &= test_coverage_query()
    
    print(f"\n{'='*50}")
    if success:
        print("✓ Tous les tests sont passés avec succès!")
    else:
        print("✗ Des problèmes ont été détectés.")
        
    print("="*50)
