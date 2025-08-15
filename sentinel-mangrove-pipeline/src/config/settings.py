import os
from pathlib import Path
from functools import lru_cache

try:
    # Charge .env si présent
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    # Postgres
    PG_HOST = os.getenv("PGHOST", "localhost")
    PG_PORT = int(os.getenv("PGPORT", "5432"))
    PG_DB = os.getenv("PGDATABASE", "global_mangrove_dataset")
    PG_USER = os.getenv("PGUSER", "postgres")
    PG_PASSWORD = os.getenv("PGPASSWORD", "")
    PG_SCHEMA = os.getenv("PGSCHEMA", "public")
    PG_TABLE = os.getenv("PGTABLE", "gmw_2016_v2")

    # Sentinel Hub
    SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "")
    SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "")

    # Téléchargement / traitement
    PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", "2560"))
    MAX_PATCHES = int(os.getenv("MAX_PATCHES", "1"))
    TIME_INTERVAL = os.getenv("TIME_INTERVAL", "2024-06-01/2025-06-10")
    MAX_CLOUD_COVER = int(os.getenv("MAX_CLOUD_COVER", "20"))
    IMAGE_RESOLUTION = int(os.getenv("IMAGE_RESOLUTION", "10"))

    # Améliorations
    ENHANCEMENT_METHOD = os.getenv("ENHANCEMENT_METHOD", "gamma")
    GAMMA_VALUE = float(os.getenv("GAMMA_VALUE", "0.9"))
    CLIP_VALUE = float(os.getenv("CLIP_VALUE", "2.2"))

    # Répertoires
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/output"))
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "data/temp"))

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        )

    def init_dirs(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

@lru_cache
def settings() -> Config:
    cfg = Config()
    cfg.init_dirs()
    return cfg