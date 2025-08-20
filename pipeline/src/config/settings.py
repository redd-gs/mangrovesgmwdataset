import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Attributs de classe (compatibilité tests accédant directement à Config.PG_USER, etc.)
    PG_HOST = os.getenv("PGHOST", "localhost")
    PG_PORT = int(os.getenv("PGPORT", "5432"))
    PG_DB = os.getenv("PGDATABASE", "global_mangrove_dataset_2016")
    PG_USER = os.getenv("PGUSER", "postgres")
    PG_PASSWORD = os.getenv("PGPASSWORD", "mangrovesondra")
    PG_SCHEMA = os.getenv("PGSCHEMA", "public")
    PG_TABLE = os.getenv("PGTABLE", "gmw_2016_v2")

    SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "296047b6-fdf8-4cf1-b5b3-25bc57cda004")
    SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "eAx3zVObhObgW6Om9t7PY5TsP6J0GD3b")
    SH_INSTANCE_ID = os.getenv("SH_INSTANCE_ID", "975be0e1-6eed-4cf0-ab03-cdb6722aab80")

    TIME_INTERVAL = os.getenv("TIME_INTERVAL", "2024-06-01/2025-06-10")
    MAX_CLOUD_COVER = int(os.getenv("MAX_CLOUD_COVER", "20"))
    IMAGE_RESOLUTION = int(os.getenv("IMAGE_RESOLUTION", "10"))
    PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", "2048"))
    MAX_PATCHES = int(os.getenv("MAX_PATCHES", "10"))

    ENHANCEMENT_METHOD = os.getenv("ENHANCEMENT_METHOD", "gamma")
    GAMMA_VALUE = float(os.getenv("GAMMA_VALUE", "0.9"))
    CLIP_VALUE = float(os.getenv("CLIP_VALUE", "2.2"))

    def __init__(self):
        # Postgres
        self.PG_HOST = os.getenv("PGHOST", "localhost")
        self.PG_PORT = int(os.getenv("PGPORT", "5432"))
        self.PG_DB = os.getenv("PGDATABASE", "global_mangrove_dataset_2016")
        self.PG_USER = os.getenv("PGUSER", "postgres")
        self.PG_PASSWORD = os.getenv("PGPASSWORD", "mangrovesondra")
        self.PG_SCHEMA = os.getenv("PGSCHEMA", "public")
        self.PG_TABLE = os.getenv("PGTABLE", "gmw_2016_v2")

        # Sentinel Hub
        self.SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "296047b6-fdf8-4cf1-b5b3-25bc57cda004")
        self.SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "eAx3zVObhObgW6Om9t7PY5TsP6J0GD3b")
        self.SH_INSTANCE_ID = os.getenv("SH_INSTANCE_ID", "975be0e1-6eed-4cf0-ab03-cdb6722aab80")  # optionnel

        # Traitement
        self.TIME_INTERVAL = os.getenv("TIME_INTERVAL", "2024-06-01/2025-06-10")
        self.MAX_CLOUD_COVER = int(os.getenv("MAX_CLOUD_COVER", "20"))
        self.IMAGE_RESOLUTION = int(os.getenv("IMAGE_RESOLUTION", "10"))
        self.PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", "512"))
        self.MAX_PATCHES = int(os.getenv("MAX_PATCHES", "10"))

        # Améliorations
        self.ENHANCEMENT_METHOD = os.getenv("ENHANCEMENT_METHOD", "gamma")
        self.GAMMA_VALUE = float(os.getenv("GAMMA_VALUE", "0.9"))
        self.CLIP_VALUE = float(os.getenv("CLIP_VALUE", "2.2"))

        # Répertoires
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/output"))
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", "data/temp"))
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_dirs(self):
        for d in (self.OUTPUT_DIR, self.TEMP_DIR):
            if d.exists() and not d.is_dir():
                raise RuntimeError(f"[ERREUR] {d} existe mais n'est pas un dossier. Supprime ce fichier.")
            d.mkdir(parents=True, exist_ok=True)

    @property
    def time_interval_tuple(self):
        """Retourne (start, end) en chaînes YYYY-MM-DD à partir de TIME_INTERVAL."""
        raw = self.TIME_INTERVAL.strip()
        if "/" in raw:
            a, b = raw.split("/", 1)
        else:
            a = b = raw
        return a.strip(), b.strip()
            
    @property
    def pg_dsn(self):
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

# Configs pour d'autres bases de données
class GmwV3Config(Config):
    def __init__(self):
        super().__init__()
        self.PG_DB = "gmw_v3"

class EstuarineMangrovesConfig(Config):
    def __init__(self):
        super().__init__()
        self.PG_DB = "estuarine_mangroves"

class MarineMangrovesConfig(Config):
    def __init__(self):
        super().__init__()
        self.PG_DB = "marine_mangroves"

# Utilitaire pour choisir la config selon le nom de la base
def get_config(db_name: str | None = None) -> Config:
    if not db_name:
        return Config()
    if db_name == "gmw_v3":
        return GmwV3Config()
    if db_name == "estuarine_mangroves":
        return EstuarineMangrovesConfig()
    if db_name == "marine_mangroves":
        return MarineMangrovesConfig()
    # Si un nom arbitraire est fourni, on retourne une Config générique avec ce nom
    cfg = Config()
    cfg.PG_DB = db_name
    return cfg
    

@lru_cache
def settings(db_name: str | None = None) -> Config:
    """Retourne une configuration (mise en cache par nom de base).

    Exemple d'utilisation:
        from src.config.settings import settings
        cfg = settings("gmw_v3")
    """
    return get_config(db_name)