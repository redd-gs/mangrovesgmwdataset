import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

REQUIRED_COLUMNS = ["latitude", "longitude", "soc", "top_depth_cm", "bottom_depth_cm"]


def load_insitu_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def normalize_depth(df: pd.DataFrame, target_interval: Tuple[float, float]) -> pd.DataFrame:
    """Normalize SOC stock to a target depth interval.

    Assumes df has columns: soc (stock for layer), top_depth_cm, bottom_depth_cm.
    If site has multiple layers, user should aggregate prior to this function or extend logic.
    """
    top_t, bot_t = target_interval
    # Simple filter for now; future: integrate partial overlaps.
    mask = (df["top_depth_cm"] <= top_t) & (df["bottom_depth_cm"] >= bot_t)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError("No samples covering the target interval entirely. Implement partial layer integration.")
    return out
