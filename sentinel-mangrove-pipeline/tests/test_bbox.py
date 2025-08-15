import pytest
from src.processing.bbox import create_valid_bbox
from shapely.geometry import Point

def test_create_valid_bbox():
    # Test with a valid geometry
    point = Point(0, 0)
    bbox = create_valid_bbox(point, size_m=1000)
    assert bbox is not None
    assert bbox.min_x == -5.0
    assert bbox.min_y == -5.0
    assert bbox.max_x == 5.0
    assert bbox.max_y == 5.0

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
    assert bbox.min_x == 5.0
    assert bbox.min_y == 5.0
    assert bbox.max_x == 15.0
    assert bbox.max_y == 15.0

def test_create_valid_bbox_small_size():
    # Test with a very small size
    point = Point(20, 20)
    bbox = create_valid_bbox(point, size_m=1)
    assert bbox is not None
    assert bbox.min_x == 19.9995
    assert bbox.min_y == 19.9995
    assert bbox.max_x == 20.0005
    assert bbox.max_y == 20.0005