import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    def __init__(self):
        # Postgres
        self.PG_HOST = os.getenv("PGHOST", "localhost")
        self.PG_PORT = int(os.getenv("PGPORT", "5432"))
        self.PG_DB = os.getenv("PGDATABASE", "gmw_v3")
        self.PG_USER = os.getenv("PGUSER", "postgres")
        self.PG_PASSWORD = os.getenv("PGPASSWORD", "mangrovesondra")
        self.PG_SCHEMA = os.getenv("PGSCHEMA", "public")
        self.PG_TABLE = os.getenv("PGTABLE", "gmw_v3_2020_vec")

        # Sentinel Hub
        self.SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "330a1c5f-a084-43d5-aec3-055eeb473c4f")
        self.SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "0eWe5qAyNypL8ZXHHFGQKAu5pAA2zVFo")
        self.SH_INSTANCE_ID = os.getenv("SH_INSTANCE_ID", "60e05787-e8ba-473d-a743-402d54d72762")  # optionnel

        # Traitement
        self.TIME_INTERVAL = os.getenv("TIME_INTERVAL", "2022-06-01/2025-06-10")
        self.MAX_CLOUD_COVER = int(os.getenv("MAX_CLOUD_COVER", "10"))
        self.IMAGE_RESOLUTION = int(os.getenv("IMAGE_RESOLUTION", "10"))
        self.PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", "8192"))
        self.PATCH_SIZE_PX = int(os.getenv("PATCH_SIZE_PX", "512"))  # Taille en pixels
        self.MAX_PATCHES = int(os.getenv("MAX_PATCHES", "60"))
        
        # Filtres de qualité d'image
        self.MIN_VALID_PIXELS_RATIO = float(os.getenv("MIN_VALID_PIXELS_RATIO", "0.8"))  # 80% de pixels valides minimum
        self.MIN_BRIGHTNESS_THRESHOLD = float(os.getenv("MIN_BRIGHTNESS_THRESHOLD", "0.02"))  # Éviter les images trop sombres
        self.MAX_BRIGHTNESS_THRESHOLD = float(os.getenv("MAX_BRIGHTNESS_THRESHOLD", "0.95"))  # Éviter les images saturées
        self.MIN_CONTRAST_RATIO = float(os.getenv("MIN_CONTRAST_RATIO", "0.1"))  # Contraste minimum
        self.RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))  # Nombre de tentatives si image invalide

        # Améliorations
        self.ENHANCEMENT_METHOD = os.getenv("ENHANCEMENT_METHOD", "gamma")
        self.GAMMA_VALUE = float(os.getenv("GAMMA_VALUE", "0.9"))
        self.CLIP_VALUE = float(os.getenv("CLIP_VALUE", "2.2"))

        # Répertoires - utiliser des chemins absolus vers le dossier pipeline/data
        base_dir = Path(__file__).parent.parent.parent  # Remonte de src/config vers pipeline
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(base_dir / "data" / "sentinel_2" / "output")))
        self.BANDS_DIR = Path(os.getenv("BANDS_DIR", str(base_dir / "data" / "sentinel_2" / "bands")))
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.BANDS_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_dirs(self):
        for d in (self.OUTPUT_DIR, self.BANDS_DIR):
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
        self.PG_TABLE = "gmw_v3_2020_vec"

class EstuarineMangrovesConfig(Config):
    def __init__(self):
        super().__init__()
        self.PG_DB = "estuarine_mangroves"
        self.PG_TABLE = "estuarine_mangroves"

class MarineMangrovesConfig(Config):
    def __init__(self):
        super().__init__()
        self.PG_DB = "marine_mangroves"
        self.PG_TABLE = "marine_mangroves"

# Utilitaire pour choisir la config selon le nom de la base
def get_config(db_name: Optional[str] = None) -> Config:
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
def settings_s2(db_name: Optional[str] = None) -> Config:
    """Retourne une configuration (mise en cache par nom de base).

    Exemple d'utilisation:
        from src.config.settings import settings
        cfg = settings("gmw_v3")
    """
    return get_config(db_name)