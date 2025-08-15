import os
from sentinelhub import SHConfig

def get_sentinel_config():
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    config.sh_base_url = "https://services.sentinel-hub.com"
    config.sh_token_url = "https://services.sentinel-hub.com/oauth/token"
    
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Please set the SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.")
    
    return config