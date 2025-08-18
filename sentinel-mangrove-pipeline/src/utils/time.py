from datetime import datetime
from typing import Tuple

def format_time_interval(raw: str) -> Tuple[str, str]:
    """
    Accepte 'YYYY-MM-DD/YYYY-MM-DD' ou 'YYYY-MM-DD'.
    Retourne (start_iso, end_iso) au format ISO 'YYYY-MM-DD'.
    """
    raw = raw.strip()
    if "/" in raw:
        start, end = raw.split("/", 1)
    else:
        start = end = raw
    fmt = "%Y-%m-%d"
    # Validation
    datetime.strptime(start.strip(), fmt)
    datetime.strptime(end.strip(), fmt)
    return start.strip(), end.strip()