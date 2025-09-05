import os
from sentinelhub import SHConfig
from config.settings_s2 import settings_s2

def get_sentinel_config():
    """Construit une config Sentinel Hub.

    Fallback: si variables d'environnement absentes on utilise les valeurs du fichier settings.
    (Utile pour les tests locaux.)
    """
    cfg = settings_s2()
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID") or cfg.SH_CLIENT_ID
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET") or cfg.SH_CLIENT_SECRET
    if hasattr(cfg, 'SH_INSTANCE_ID') and cfg.SH_INSTANCE_ID:
        config.instance_id = cfg.SH_INSTANCE_ID  # optionnel selon version sentinelhub
    config.sh_base_url = "https://services.sentinel-hub.com"
    config.sh_token_url = "https://services.sentinel-hub.com/oauth/token"
    # Ne pas lever d'exception ici pour permettre aux tests de passer/être sautés plus haut.
    return config

# Compatibilité tests: certains tests appellent sh_config()
def sh_config():
    return get_sentinel_config()