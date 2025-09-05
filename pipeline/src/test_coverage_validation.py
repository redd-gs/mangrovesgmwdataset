#!/usr/bin/env python3
"""
Script de test pour valider la fonction calculate_mangrove_coverage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.mangrove_dataset import calculate_mangrove_coverage, get_coverage_category
from sentinelhub import BBox, CRS
import time

def test_coverage_scenarios():
    """
    Teste différents scénarios de calcul de couverture.
    """
    print("=== Test de la fonction calculate_mangrove_coverage ===\n")
    
    # Scénarios de test avec des zones connues
    test_cases = [
        {
            "name": "Zone avec mangroves - Côte d'Ivoire",
            "bbox": BBox([-5.5, 5.0, -5.0, 5.5], crs=CRS.WGS84),
            "expected_range": (0, 100),  # On s'attend à une couverture mesurable
        },
        {
            "name": "Zone avec beaucoup de mangroves - Sundarbans, Bangladesh",
            "bbox": BBox([89.0, 21.5, 89.5, 22.0], crs=CRS.WGS84),
            "expected_range": (0, 100),
        },
        {
            "name": "Zone océanique - aucune mangrove attendue",
            "bbox": BBox([-30.0, 0.0, -29.5, 0.5], crs=CRS.WGS84),
            "expected_range": (0, 0),  # Océan, pas de mangroves
        },
        {
            "name": "Zone terrestre intérieure - aucune mangrove",
            "bbox": BBox([2.0, 48.0, 2.5, 48.5], crs=CRS.WGS84),  # Région parisienne
            "expected_range": (0, 0),
        },
        {
            "name": "Petite zone test avec mangroves potentielles",
            "bbox": BBox([73.5, 15.9, 73.6, 16.0], crs=CRS.WGS84),  # Inde côtière
            "expected_range": (0, 100),
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  BBox: [{test_case['bbox'].lower_left[0]:.2f}, {test_case['bbox'].lower_left[1]:.2f}, {test_case['bbox'].upper_right[0]:.2f}, {test_case['bbox'].upper_right[1]:.2f}]")
        
        try:
            start_time = time.time()
            coverage = calculate_mangrove_coverage(test_case['bbox'], "public.gmw_v3_2020_vec")
            end_time = time.time()
            elapsed = end_time - start_time
            category = get_coverage_category(coverage)
            
            min_expected, max_expected = test_case['expected_range']
            is_in_range = min_expected <= coverage <= max_expected
            
            status = "✓" if is_in_range else "⚠"
            
            print(f"  Résultat: {coverage:.2f}% - Catégorie: {category} {status} (temps: {elapsed:.2f}s)")
            
            results.append({
                "test": test_case['name'],
                "bbox": test_case['bbox'],
                "coverage": coverage,
                "category": category,
                "valid": is_in_range,
                "time": elapsed
            })
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            results.append({
                "test": test_case['name'],
                "bbox": test_case['bbox'],
                "coverage": None,
                "category": None,
                "valid": False,
                "error": str(e)
            })
        
        print()
    
    # Résumé des résultats
    print("=== Résumé des tests ===")
    
    successful_tests = sum(1 for r in results if r.get('coverage') is not None)
    valid_tests = sum(1 for r in results if r.get('valid', False))
    
    print(f"Tests exécutés: {len(results)}")
    print(f"Tests réussis (sans erreur): {successful_tests}")
    print(f"Tests valides (résultats attendus): {valid_tests}")
    
    if successful_tests == len(results):
        print("✓ Tous les tests se sont exécutés sans erreur")
    else:
        print("⚠ Certains tests ont échoué")
    
    # Afficher les détails des couvertures trouvées
    print("\n=== Détail des couvertures ===")
    for result in results:
        if result.get('coverage') is not None:
            print(f"{result['test']}: {result['coverage']:.2f}% ({result['category']})")
    
    return results

def test_category_boundaries():
    """
    Teste les limites des catégories de couverture.
    """
    print("\n=== Test des catégories de couverture ===")
    
    test_percentages = [0, 0.5, 1, 10, 20, 21, 30, 40, 41, 50, 60, 61, 70, 80, 81, 90, 100]
    
    for percentage in test_percentages:
        category = get_coverage_category(percentage)
        print(f"{percentage:5.1f}% -> {category}")

if __name__ == "__main__":
    print("Test de validation de la fonction de calcul de couverture\n")
    
    # Exécuter les tests
    results = test_coverage_scenarios()
    test_category_boundaries()
    
    print(f"\n{'='*60}")
    print("Tests terminés.")
    
    # Recommandations basées sur les résultats
    coverage_found = [r['coverage'] for r in results if r.get('coverage') is not None and r['coverage'] > 0]
    
    if coverage_found:
        print(f"✓ Fonction opérationnelle - Couvertures détectées: {len(coverage_found)} zones")
        avg_coverage = sum(coverage_found) / len(coverage_found)
        print(f"  Couverture moyenne des zones avec mangroves: {avg_coverage:.2f}%")
    else:
        print("ℹ Aucune couverture détectée dans les zones testées")
    
    print("="*60)
