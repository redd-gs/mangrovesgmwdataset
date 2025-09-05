"""
Ce module gère la création d'un jeu de données d'images de mangroves
en se basant sur des catégories de densité de couverture.
"""
from typing import Dict, List, Tuple
import os
import random
from shapely.geometry import box
from sqlalchemy import text
from sentinelhub import BBox, CRS

from config.context import get_engine, get_sh_config
from sentinel.download_s2 import run_download
from database.gmw_v3 import fetch_random_polygons

# Définition des catégories de couverture de mangrove
COVERAGE_CATEGORIES = {
    "no_mangroves": (0, 0),
    "1_20_percent": (1, 20),
    "21_40_percent": (21, 40),
    "41_60_percent": (41, 60),
    "61_80_percent": (61, 80),
    "more_than_80_percent": (81, 100),
}

def get_coverage_category(percentage: float) -> str:
    """Retourne le nom de la catégorie pour un pourcentage de couverture donné."""
    if percentage == 0:
        return "no_mangroves"
    for name, (min_val, max_val) in COVERAGE_CATEGORIES.items():
        if min_val <= percentage <= max_val:
            return name
    # Si aucune catégorie ne correspond, utiliser la catégorie la plus proche
    if percentage > 100:
        return "more_than_80_percent"
    else:
        return "no_mangroves"  # Fallback sécurisé

def validate_database_connection() -> bool:
    """
    Valide la connexion à la base de données et les extensions PostGIS.
    
    Returns:
        bool: True si la connexion et PostGIS fonctionnent, False sinon
    """
    try:
        engine = get_engine()
        
        # Test simple de PostGIS
        sql = text("SELECT ST_AsText(ST_Point(0, 0)) as point_wkt;")
        
        with engine.connect() as conn:
            result = conn.execute(sql).scalar_one_or_none()
            
        if result and result == "POINT(0 0)":
            print("[INFO] Connexion à la base de données et PostGIS validées")
            return True
        else:
            print("[ERROR] Problème avec PostGIS")
            return False
            
    except Exception as e:
        print(f"[ERROR] Impossible de se connecter à la base de données: {e}")
        return False

def test_srid_operations() -> bool:
    """
    Teste les opérations SRID pour s'assurer qu'elles fonctionnent correctement.
    
    Returns:
        bool: True si les opérations SRID fonctionnent, False sinon
    """
    try:
        engine = get_engine()
        
        # Test de transformation SRID
        sql = text("""
            SELECT ST_AsText(
                ST_Transform(
                    ST_SetSRID(ST_Point(5.5, 5.9), 4326), 
                    3857
                )
            ) as transformed_point;
        """)
        
        with engine.connect() as conn:
            result = conn.execute(sql).scalar_one_or_none()
            
        if result:
            print(f"[INFO] Test SRID réussi: {result}")
            return True
        else:
            print("[ERROR] Échec du test SRID")
            return False
            
    except Exception as e:
        print(f"[ERROR] Impossible de tester les opérations SRID: {e}")
        return False

def test_coverage_calculation(bbox: BBox, gmw_table: str) -> Dict:
    """
    Fonction de test pour diagnostiquer le calcul de couverture.
    Retourne des informations détaillées sur le calcul.
    """
    try:
        engine = get_engine()
        
        # Créer la géométrie de la bbox
        bbox_wkt = f"POLYGON(({bbox.min_x} {bbox.min_y}, {bbox.max_x} {bbox.min_y}, {bbox.max_x} {bbox.max_y}, {bbox.min_x} {bbox.max_y}, {bbox.min_x} {bbox.min_y}))"
        
        # Requête de diagnostic
        sql = text(f"""
            WITH bbox_geom AS (
                SELECT ST_Transform(ST_GeomFromText(:bbox_wkt, 4326), 3857) as geom
            ),
            bbox_area AS (
                SELECT ST_Area(geom) as total_area, geom FROM bbox_geom
            ),
            mangrove_stats AS (
                SELECT 
                    COUNT(*) as polygon_count,
                    SUM(ST_Area(ST_Transform(t.geom, 3857))) as total_mangrove_area,
                    SUM(ST_Area(ST_Intersection(ST_Transform(t.geom, 3857), ba.geom))) as intersection_area
                FROM "{gmw_table.split('.')[0]}"."{gmw_table.split('.')[1]}" t
                CROSS JOIN bbox_area ba
                WHERE ST_Intersects(ST_Transform(t.geom, 3857), ba.geom)
                  AND ST_IsValid(t.geom)
                  AND NOT ST_IsEmpty(t.geom)
            )
            SELECT
                ba.total_area,
                COALESCE(ms.polygon_count, 0) as polygon_count,
                COALESCE(ms.total_mangrove_area, 0) as total_mangrove_area,
                COALESCE(ms.intersection_area, 0) as intersection_area,
                COALESCE(ms.intersection_area / ba.total_area * 100, 0) as coverage_percentage
            FROM bbox_area ba
            LEFT JOIN mangrove_stats ms ON true;
        """)

        with engine.connect() as conn:
            result = conn.execute(sql, {"bbox_wkt": bbox_wkt}).fetchone()
            
        if result:
            return {
                "bbox_area_m2": float(result[0]),
                "mangrove_polygons_count": int(result[1]),
                "total_mangrove_area_m2": float(result[2]),
                "intersection_area_m2": float(result[3]),
                "coverage_percentage": float(result[4]),
                "bbox_coordinates": (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
            }
        else:
            return {"error": "Aucun résultat retourné par la requête"}
            
    except Exception as e:
        return {"error": str(e)}

def calculate_mangrove_coverage(bbox, gmw_table: str = "public.gmw_v3_2020_vec") -> float:
    """
    Version optimisée du calcul de couverture de mangroves.
    Utilise automatiquement la méthode batch pour un seul élément.
    """
    try:
        # Import de la version optimisée
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from optimized_coverage import calculate_mangrove_coverage_batch
        
        # Utiliser la méthode batch même pour un seul élément (plus efficace)
        coverages = calculate_mangrove_coverage_batch([bbox], gmw_table)
        return coverages[0] if coverages else 0.0
        
    except ImportError:
        print("[WARNING] Module optimisé non disponible, utilisation de la méthode standard...")
        # Fallback vers l'ancienne méthode si l'import échoue
        return calculate_mangrove_coverage_legacy(bbox, gmw_table)

def calculate_mangrove_coverage_legacy(bbox, gmw_table: str) -> float:
    """
    Calcule le pourcentage de la BBox qui est couverte par des mangroves de la table GMW.
    """
    try:
        engine = get_engine()
        
        # Créer la géométrie de la bbox en WGS84 (EPSG:4326)
        bbox_poly = box(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
        
        # Construire la géométrie WKT (la méthode la plus simple maintenant que le SRID est correct)
        bbox_wkt = f"POLYGON(({bbox.min_x} {bbox.min_y}, {bbox.max_x} {bbox.min_y}, {bbox.max_x} {bbox.max_y}, {bbox.min_x} {bbox.max_y}, {bbox.min_x} {bbox.min_y}))"

        # Requête SQL pour calculer la couverture
        sql = text(f"""
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
                FROM "{gmw_table.split('.')[0]}"."{gmw_table.split('.')[1]}" t
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

        with engine.connect() as conn:
            result = conn.execute(sql, {"bbox_wkt": bbox_wkt}).scalar_one_or_none()

        coverage = result if result is not None else 0.0
        
        # S'assurer que le résultat est dans la plage [0, 100]
        coverage = max(0.0, min(100.0, coverage))
        
        return coverage
        
    except Exception as e:
        print(f"[WARNING] Impossible de calculer la couverture de mangroves: {e}")
        return 0.0

def generate_dataset(
    output_dir: str,
    images_per_category: int,
    patch_size_px: int,
    patch_size_m: int,
    gmw_table: str
):
    """
    Génère le jeu de données d'images de manière équilibrée.
    """
    print("[INFO] Validation de la connexion à la base de données...")
    if not validate_database_connection():
        print("[ERROR] Impossible de continuer sans connexion à la base de données")
        return
    
    print("[INFO] Test des opérations SRID...")
    if not test_srid_operations():
        print("[ERROR] Problème avec les opérations SRID")
        return
    
    # Initialiser le compteur pour chaque catégorie
    category_counts = {name: 0 for name in COVERAGE_CATEGORIES}

    # Créer les dossiers de sortie pour chaque catégorie
    for category_name in COVERAGE_CATEGORIES:
        os.makedirs(os.path.join(output_dir, category_name), exist_ok=True)

    total_images_needed = len(COVERAGE_CATEGORIES) * images_per_category
    
    # On va chercher des polygones de mangroves pour générer des BBox candidates
    # On en prend plus que nécessaire pour avoir de la variété
    print("[INFO] Récupération des polygones de mangroves pour générer des zones candidates...")
    mangrove_polygons = fetch_random_polygons(limit=total_images_needed * 2)
    
    # Ajoutons aussi des zones aléatoires sur la planète pour la catégorie "no_mangroves"
    # pour assurer la diversité (zones côtières sans mangroves, terres, etc.)
    random_bboxes = []
    for _ in range(images_per_category * 2):
        # Génère des BBox aléatoires sur les côtes du monde
        lon = random.uniform(-180, 180)
        lat = random.uniform(-30, 30) # Focus sur la ceinture tropicale
        bbox = BBox.get_transform_bbox(BBox((lon, lat, lon, lat), crs=CRS.WGS84), CRS.UTM_NORTH if lat >= 0 else CRS.UTM_SOUTH, patch_size_m)
        random_bboxes.append(bbox)

    candidate_bboxes = random_bboxes
    for poly in mangrove_polygons:
        # Centre la BBox sur un point du polygone de mangrove
        center = poly.representative_point()
        utm_crs = CRS.UTM_NORTH if center.y >= 0 else CRS.UTM_SOUTH
        bbox = BBox.get_transform_bbox(BBox((center.x, center.y, center.x, center.y), crs=CRS.WGS84), utm_crs, patch_size_m)
        candidate_bboxes.append(bbox)
    
    random.shuffle(candidate_bboxes)
    print(f"[INFO] {len(candidate_bboxes)} zones candidates générées. Début du traitement...")

    processed_images = 0
    for i, bbox in enumerate(candidate_bboxes):
        
        # Vérifier si toutes les catégories sont complètes
        if all(count >= images_per_category for count in category_counts.values()):
            print("[SUCCESS] Toutes les catégories ont atteint le nombre d'images souhaité.")
            break

        # Calculer la couverture
        coverage = calculate_mangrove_coverage(bbox, gmw_table)
        category = get_coverage_category(coverage)

        # Afficher des informations de diagnostic pour les premières images ou en cas d'erreur
        if i < 5 or coverage == 0.0:  # Debug pour les 5 premières ou quand pas de mangroves
            debug_info = test_coverage_calculation(bbox, gmw_table)
            if "error" in debug_info:
                print(f"[DEBUG] Erreur de diagnostic: {debug_info['error']}")
            else:
                print(f"[DEBUG] Zone {i+1}: BBox area={debug_info['bbox_area_m2']:.0f}m², "
                      f"Polygones mangroves={debug_info['mangrove_polygons_count']}, "
                      f"Intersection={debug_info['intersection_area_m2']:.0f}m², "
                      f"Couverture={debug_info['coverage_percentage']:.2f}%")

        if category and category_counts[category] < images_per_category:
            print(f"[INFO] Zone {i+1}/{len(candidate_bboxes)}: Couverture de {coverage:.2f}%. Catégorie: {category}. Téléchargement...")
            
            # Nom de fichier unique
            filename = f"{category}_{category_counts[category] + 1}.png"
            category_path = os.path.join(output_dir, category)
            output_path = os.path.join(category_path, filename)

            # Télécharger l'image
            try:
                config = get_sh_config()
                run_download(bbox, config, output_path, patch_size_px)
                category_counts[category] += 1
                processed_images += 1
                print(f"[SUCCESS] Image sauvegardée dans {output_path}")
                print(f"[INFO] Image classée dans la catégorie '{category}' ({coverage:.2f}% de couverture)")
                print(f"[STATS] Progression: {category_counts}")

            except Exception as e:
                print(f"[ERROR] Impossible de télécharger l'image pour la bbox {bbox}. Erreur: {e}")
        else:
            if category:
                print(f"[INFO] Zone {i+1}: Couverture {coverage:.2f}% (catégorie {category}) - catégorie déjà complète, ignore.")
            else:
                print(f"[INFO] Zone {i+1}: Couverture {coverage:.2f}% - ne correspond à aucune catégorie, ignore.")
    
    print("\n--- Rapport final de génération du dataset ---")
    for category, count in category_counts.items():
        print(f"- Catégorie '{category}': {count}/{images_per_category} images.")
    print(f"Total d'images générées: {processed_images}")
    print("--------------------------------------------")
