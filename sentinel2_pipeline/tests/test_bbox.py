import pytest
from src.processing.bbox import create_valid_bbox
from shapely.geometry import Point

def test_create_valid_bbox():
    # Test with a valid geometry
    point = Point(0, 0)
    bbox = create_valid_bbox(point, size_m=1000)
    assert bbox is not None
    # Les valeurs ne sont pas exactement +/-5 degrés car on convertit 1000 m en EPSG:3857 puis on reprojette.
    # On vérifie juste une symétrie approximative autour de 0 et une amplitude cohérente (~0.009°).
    assert pytest.approx(bbox.min_x, rel=1e-3) == -bbox.max_x
    assert pytest.approx(bbox.min_y, rel=1e-3) == -bbox.max_y
    assert 0.008 < (bbox.max_x - bbox.min_x) < 0.010

def test_create_valid_bbox_empty_geometry():
    # Test with an empty geometry
    point = Point()
    bbox = create_valid_bbox(point, size_m=1000)
    assert bbox is None

def test_create_valid_bbox_large_size():
    # Test with a larger size
    point = Point(10, 10)
    bbox = create_valid_bbox(point, size_m=10000)
    assert bbox is not None
    # Taille 10000 m produit ~0.089° en longitude à cette latitude; on teste cohérence
    assert 9.9 < bbox.min_x < 10.0
    assert 10.0 < bbox.max_x < 10.1
    assert 9.9 < bbox.min_y < 10.0
    assert 10.0 < bbox.max_y < 10.1

def test_create_valid_bbox_small_size():
    # Test with a very small size
    point = Point(20, 20)
    bbox = create_valid_bbox(point, size_m=1)
    assert bbox is not None
    # Très petite taille ~1 m -> delta ~1e-4°
    assert 19.9998 < bbox.min_x < 20.0
    assert 20.0 < bbox.max_x < 20.0002
    assert 19.9998 < bbox.min_y < 20.0
    assert 20.0 < bbox.max_y < 20.0002