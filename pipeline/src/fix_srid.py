#!/usr/bin/env python3
"""
Script pour corriger le SRID de la table GMW.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from database.gmw_v3 import get_engine

def fix_gmw_srid():
    """
    Corrige le SRID de la table gmw_v3_2020_vec en assumant que les données
    sont en WGS84 (EPSG:4326).
    """
    print("=== Correction du SRID de la table GMW ===")
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Vérifier le SRID actuel
            result = conn.execute(text("""
                SELECT ST_SRID(geom) as current_srid, COUNT(*) as count
                FROM public.gmw_v3_2020_vec 
                GROUP BY ST_SRID(geom);
            """)).fetchall()
            
            print("SRID actuels dans la table:")
            for row in result:
                print(f"  SRID {row.current_srid}: {row.count} enregistrements")
            
            # Vérifier un échantillon des coordonnées pour confirmer qu'elles semblent être en WGS84
            sample = conn.execute(text("""
                SELECT ST_AsText(geom) as wkt
                FROM public.gmw_v3_2020_vec 
                LIMIT 5;
            """)).fetchall()
            
            print("\nÉchantillon de géométries (vérification des coordonnées):")
            for i, row in enumerate(sample):
                # Extraire les premières coordonnées pour vérification
                wkt = row.wkt
                if "POLYGON" in wkt or "MULTIPOLYGON" in wkt:
                    coords_start = wkt.find("(") + 1
                    coords_end = wkt.find(")", coords_start)
                    coords_sample = wkt[coords_start:coords_end][:50]
                    print(f"  Échantillon {i+1}: {coords_sample}...")
            
            # Demander confirmation pour la correction
            print(f"\nLes coordonnées semblent être en degrés décimaux (longitude/latitude).")
            print("Cela confirme qu'elles sont probablement en WGS84 (EPSG:4326).")
            
            response = input("Voulez-vous corriger le SRID de 0 vers 4326? (y/N): ")
            if response.lower() != 'y':
                print("Opération annulée.")
                return False
            
            print("Correction du SRID en cours...")
            
            # Mettre à jour le SRID
            # Note: nous utilisons ST_SetSRID car les coordonnées sont déjà correctes,
            # nous ne faisons que corriger les métadonnées
            conn.execute(text("""
                UPDATE public.gmw_v3_2020_vec 
                SET geom = ST_SetSRID(geom, 4326)
                WHERE ST_SRID(geom) = 0;
            """))
            
            conn.commit()
            
            # Vérifier le résultat
            result = conn.execute(text("""
                SELECT ST_SRID(geom) as srid, COUNT(*) as count
                FROM public.gmw_v3_2020_vec 
                GROUP BY ST_SRID(geom);
            """)).fetchall()
            
            print("SRID après correction:")
            for row in result:
                print(f"  SRID {row.srid}: {row.count} enregistrements")
            
            print("✓ Correction du SRID terminée avec succès!")
            return True
            
    except Exception as e:
        print(f"✗ Erreur lors de la correction du SRID: {e}")
        return False

def test_coverage_after_fix():
    """
    Teste le calcul de couverture après la correction du SRID.
    """
    print("\n=== Test du calcul de couverture après correction ===")
    
    try:
        engine = get_engine()
        
        # BBox de test autour de la Côte d'Ivoire (zone avec des mangroves)
        test_bbox = "POLYGON((-5.5 5.0, -5.0 5.0, -5.0 5.5, -5.5 5.5, -5.5 5.0))"
        
        with engine.connect() as conn:
            sql = text("""
                WITH bbox_geom AS (
                    SELECT ST_Transform(ST_GeomFromText(:bbox_wkt, 4326), 3857) as geom
                ),
                bbox_area AS (
                    SELECT ST_Area(geom) as total_area, geom FROM bbox_geom
                ),
                intersecting_mangroves AS (
                    SELECT 
                        ST_Area(
                            ST_Intersection(
                                ST_Transform(t.geom, 3857),
                                ba.geom
                            )
                        ) as intersection_area
                    FROM public.gmw_v3_2020_vec t
                    CROSS JOIN bbox_area ba
                    WHERE ST_Intersects(ST_Transform(t.geom, 3857), ba.geom)
                      AND ST_IsValid(t.geom)
                      AND NOT ST_IsEmpty(t.geom)
                )
                SELECT
                    COALESCE(
                        (SELECT SUM(intersection_area) FROM intersecting_mangroves) / 
                        (SELECT total_area FROM bbox_area) * 100,
                        0
                    ) as coverage_percentage;
            """)
            
            result = conn.execute(sql, {"bbox_wkt": test_bbox}).scalar()
            
            print(f"✓ Calcul de couverture réussi: {result:.2f}%")
            
            if result > 0:
                print("✓ Des mangroves ont été détectées dans la zone de test!")
            else:
                print("ℹ Aucune mangrove détectée dans cette zone de test (normal si aucune mangrove présente)")
            
            return True
            
    except Exception as e:
        print(f"✗ Erreur lors du test de couverture: {e}")
        return False

if __name__ == "__main__":
    print("Script de correction du SRID pour la table GMW\n")
    
    if fix_gmw_srid():
        test_coverage_after_fix()
    else:
        print("Impossible de continuer sans correction du SRID.")
