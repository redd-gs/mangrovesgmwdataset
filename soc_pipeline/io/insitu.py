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

def group_by_site(df: pd.DataFrame) -> pd.DataFrame:
        """Group data by unique latitude and longitude combinations (sites)."""
        grouped = df.groupby(['latitude', 'longitude']).agg({
            'soc': list,  # Collect SOC values for each layer
            'top_depth_cm': list,
            'bottom_depth_cm': list
        }).reset_index()
        return grouped


def aggregate_soc_by_centroid(grouped_df: pd.DataFrame, target_depth: float = 30.0) -> pd.DataFrame:
    """Aggregate SOC stock per site using centroid rules up to target depth.
        
        For each site, calculate the centroid depth for each layer and aggregate SOC
        proportionally up to the target depth (default 30 cm, per IPCC guidelines).
        """
    aggregated = []
    for _, row in grouped_df.iterrows():
        soc_values = row['soc']
        top_depths = row['top_depth_cm']
        bottom_depths = row['bottom_depth_cm']
            
        total_soc = 0.0
        cumulative_depth = 0.0
            
        for soc, top, bot in zip(soc_values, top_depths, bottom_depths):
            layer_thickness = bot - top
            centroid_depth = (top + bot) / 2
                
            if cumulative_depth + layer_thickness <= target_depth:
                total_soc += soc
                cumulative_depth += layer_thickness
            else:
                remaining_depth = target_depth - cumulative_depth
                fraction = remaining_depth / layer_thickness
                total_soc += soc * fraction
                cumulative_depth = target_depth
                break
            
        aggregated.append({
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'aggregated_soc': total_soc
            })
        
    return pd.DataFrame(aggregated)
        