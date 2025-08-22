from __future__ import annotations
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from typing import List, Iterator, Tuple, Optional
from sklearn.model_selection import KFold


def make_spatial_blocks(gdf: gpd.GeoDataFrame, n_x: int, n_y: int) -> List[gpd.GeoDataFrame]:
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.linspace(minx, maxx, n_x + 1)
    ys = np.linspace(miny, maxy, n_y + 1)
    blocks = []
    for i in range(n_x):
        for j in range(n_y):
            geom = box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            subset = gdf[gdf.geometry.within(geom)].copy()
            if not subset.empty:
                subset["_block_id"] = f"b{i}_{j}"
                blocks.append(subset)
    return blocks


def spatial_block_cv_indices(blocks: List[gpd.GeoDataFrame]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    # Each block as test once; rest train
    all_idx = np.concatenate([b.index.values for b in blocks])
    for b in blocks:
        test_idx = b.index.values
        train_idx = all_idx[~np.isin(all_idx, test_idx)]
        yield train_idx, test_idx


def leave_group_out_indices(gdf: gpd.GeoDataFrame, group_col: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    groups = gdf[group_col].unique()
    for g in groups:
        test_idx = gdf[gdf[group_col] == g].index.values
        train_idx = gdf[gdf[group_col] != g].index.values
        yield train_idx, test_idx


def kfold_blocks(gdf: gpd.GeoDataFrame, n_splits: int = 5, random_state: Optional[int] = 42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = np.arange(len(gdf))
    for train, test in kf.split(idx):
        yield gdf.index.values[train], gdf.index.values[test]
