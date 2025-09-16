#!/usr/bin/env python3
"""
Alternative implementation using GDAL directly instead of rasterio to avoid DLL issues.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Mapping, Any
import traceback

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
except ImportError as e:
    print(f"Error importing GDAL: {e}")
    print("Please install GDAL: pip install GDAL")
    sys.exit(1)

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection


def _coerce_engine(conn_or_engine: Union[str, Engine, Connection]) -> Connection:
    """Return a live Connection from a SQLAlchemy Engine/Connection or DSN string."""
    if isinstance(conn_or_engine, Connection):
        return conn_or_engine
    if isinstance(conn_or_engine, Engine):
        return conn_or_engine.connect()
    if isinstance(conn_or_engine, str):
        eng = create_engine(conn_or_engine, future=True)
        return eng.connect()
    raise TypeError("conn_or_engine must be a SQLAlchemy Connection, Engine, or DB URL string")


def insert_tiff_with_gdal(
    conn_or_engine: Union[str, Engine, Connection],
    table: str,
    raster_column: str,
    tiff_path: str,
    extra_columns: Optional[Mapping[str, Any]] = None,
) -> None:
    """Insert a GeoTIFF using GDAL's ST_FromGDALRaster function directly.
    
    This bypasses the WKB conversion and uses PostGIS's built-in GDAL support.
    """
    # Read the file as binary data
    with open(tiff_path, 'rb') as f:
        tiff_data = f.read()
    
    conn = _coerce_engine(conn_or_engine)
    
    # Build the SQL
    cols = [raster_column]
    vals_placeholders = ["ST_FromGDALRaster(:tiff_data)"]
    params = {"tiff_data": tiff_data}
    
    if extra_columns:
        for k, v in extra_columns.items():
            cols.append(k)
            params[k] = v
            vals_placeholders.append(f":{k}")
    
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals_placeholders)})"
    
    with conn.begin():
        conn.execute(text(sql), params)


def ingest_directory_tiffs_gdal(
    conn_or_engine: Union[str, Engine, Connection],
    table: str,
    raster_column: str,
    dir_path: Union[str, os.PathLike],
    recursive: bool = True,
) -> int:
    """Insert all GeoTIFFs from a directory using GDAL directly."""
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")

    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(p.rglob(pat) if recursive else p.glob(pat))
    
    files = sorted({f.resolve() for f in files})
    if not files:
        print(f"No TIFF files found in {p}")
        return 0

    print(f"Found {len(files)} TIFF files to process...")
    
    conn = _coerce_engine(conn_or_engine)
    
    # Check if table has name/filename column
    try:
        with conn.begin() as trans:
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = :table_name AND column_name IN ('name', 'filename', 'tile')
            """), {"table_name": table.split('.')[-1]})
            available_cols = {row[0] for row in result}
    except Exception as e:
        print(f"Warning: Could not check table columns: {e}")
        available_cols = set()
    
    name_col = None
    if 'name' in available_cols:
        name_col = 'name'
    elif 'filename' in available_cols:
        name_col = 'filename'
    elif 'tile' in available_cols:
        name_col = 'tile'

    inserted = 0
    failed = 0
    
    for i, fp in enumerate(files, 1):
        try:
            extra = {}
            if name_col:
                extra[name_col] = fp.stem
            
            print(f"[{i}/{len(files)}] Processing {fp.name}...")
            insert_tiff_with_gdal(conn, table, raster_column, str(fp), extra)
            inserted += 1
            
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to insert {fp.name}: {e}")
            if failed > 10:  # Stop after too many failures
                print("Too many failures, stopping...")
                break
    
    return inserted


def main():
    """Main execution function."""
    # Use your path here - modify as needed
    path = "/Users/galex/Downloads/tt_final"
    
    if not path or not Path(path).exists():
        print(f"Path not found: {path}")
        print("Please update the 'path' variable in this script.")
        return
    
    try:
        from soc_pipeline.config.settings import Config
        cfg = Config()
        dsn = cfg.pg_dsn
        table = "public.mangrove_carbon"
        raster_col = "rast"  # Assumption: raster column is named 'rast'
        
        print(f"Connecting to database: {dsn}")
        print(f"Target table: {table}")
        print(f"Raster column: {raster_col}")
        print(f"Source directory: {path}")
        
        n = ingest_directory_tiffs_gdal(dsn, table, raster_col, path)
        print(f"\n✓ Successfully inserted {n} raster(s) into {table}.")
        
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
