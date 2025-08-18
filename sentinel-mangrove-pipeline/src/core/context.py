# Ce script crée la connexion au Sentinel Hub (API de Planet Labs) et qui nous permet ainsi de récupérer les images satellite de Sentinel-2
# Ce script permet aussi la connexion à PostGreSQL
from functools import lru_cache
from sentinelhub import SHConfig, SentinelHubSession
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from config.settings import settings

@lru_cache
def get_sh_config() -> SHConfig:
    cfg_app = settings()
    shc = SHConfig()
    shc.sh_client_id = cfg_app.SH_CLIENT_ID
    shc.sh_client_secret = cfg_app.SH_CLIENT_SECRET
    if cfg_app.SH_INSTANCE_ID:
        shc.instance_id = cfg_app.SH_INSTANCE_ID
    return shc

@lru_cache
def get_sh_session() -> SentinelHubSession:
    return SentinelHubSession(config=get_sh_config())

@lru_cache
def get_engine() -> Engine:
    return create_engine(settings().pg_dsn, pool_pre_ping=True, future=True)