import pytest
from sqlalchemy import create_engine
from src.sentinel.catalog_search import CatalogSearch
from src.sentinel.auth import sh_config
from src.config.settings import Config

@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine(f"postgresql+psycopg2://{Config.PG_USER}:{Config.PG_PASSWORD}@{Config.PG_HOST}:{Config.PG_PORT}/{Config.PG_DB}")
    yield engine
    engine.dispose()

@pytest.fixture(scope="module")
def sentinel_config():
    return sh_config()

def test_search_images(db_engine, sentinel_config):
    bbox = [102.0, -5.0, 104.0, -3.0]  # Example bounding box
    max_cloud_cover = 20  # Example max cloud cover percentage
    cs = CatalogSearch(sentinel_config)
    results = cs.search_images(bbox, max_cloud_cover)

    assert isinstance(results, list)
    assert len(results) > 0
    for image in results:
        assert 'id' in image
        assert 'properties' in image
        assert 'datetime' in image['properties']
        assert 'eo:cloud_cover' in image['properties']