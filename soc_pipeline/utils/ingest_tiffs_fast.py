#!/usr/bin/env python3
"""
Multithreaded TIFF ingestion with connection pooling for faster processing.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Mapping, Any
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError as e:
    print(f"Error importing GDAL: {e}")
    sys.exit(1)

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool


class TiffIngester:
    def __init__(self, dsn: str, table: str, raster_column: str = "rast", max_workers: int = 4):
        """Initialize the TIFF ingester with connection pooling."""
        self.table = table
        self.raster_column = raster_column
        self.max_workers = max_workers
        
        # Create engine with connection pool
        self.engine = create_engine(
            dsn,
            poolclass=QueuePool,
            pool_size=max_workers + 2,  # A few extra connections
            max_overflow=max_workers,
            pool_timeout=30,
            pool_recycle=3600  # Recycle connections every hour
        )
        
        # Stats tracking
        self.stats_lock = Lock()
        self.inserted = 0
        self.failed = 0
        self.start_time = None
    
    def insert_single_tiff(self, tiff_path: Path) -> bool:
        """Insert a single TIFF file."""
        try:
            # Read file as binary
            with open(tiff_path, 'rb') as f:
                tiff_data = f.read()
            
            # Use connection from pool
            with self.engine.connect() as conn:
                with conn.begin():
                    # Get geo info for the name (optional - could extract from TIFF metadata)
                    name = tiff_path.stem
                    
                    # Insert using ST_FromGDALRaster
                    sql = text(f"""
                        INSERT INTO {self.table} ({self.raster_column}, name) 
                        VALUES (ST_FromGDALRaster(:tiff_data), :name)
                    """)
                    
                    conn.execute(sql, {"tiff_data": tiff_data, "name": name})
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert {tiff_path.name}: {e}")
            return False
    
    def update_stats(self, success: bool):
        """Thread-safe stats update."""
        with self.stats_lock:
            if success:
                self.inserted += 1
            else:
                self.failed += 1
            
            total_processed = self.inserted + self.failed
            if total_processed % 100 == 0:  # Progress every 100 files
                elapsed = time.time() - self.start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {total_processed} processed ({self.inserted} success, {self.failed} failed) - {rate:.1f} files/sec")
    
    def ingest_directory(self, dir_path: Union[str, os.PathLike], recursive: bool = True) -> tuple[int, int]:
        """Ingest all TIFF files from a directory using multiple threads."""
        p = Path(dir_path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory not found: {p}")

        # Find all TIFF files
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        files: list[Path] = []
        for pat in patterns:
            files.extend(p.rglob(pat) if recursive else p.glob(pat))
        
        files = sorted({f.resolve() for f in files})
        if not files:
            print(f"No TIFF files found in {p}")
            return 0, 0

        print(f"Found {len(files)} TIFF files to process using {self.max_workers} threads...")
        self.start_time = time.time()
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.insert_single_tiff, file_path): file_path 
                for file_path in files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success = future.result()
                    self.update_stats(success)
                except Exception as e:
                    print(f"[ERROR] Exception processing {file_path.name}: {e}")
                    self.update_stats(False)
        
        total_time = time.time() - self.start_time
        avg_rate = (self.inserted + self.failed) / total_time if total_time > 0 else 0
        
        print(f"\n=== Final Results ===")
        print(f"Total processed: {self.inserted + self.failed}")
        print(f"Successfully inserted: {self.inserted}")
        print(f"Failed: {self.failed}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average rate: {avg_rate:.1f} files/second")
        
        return self.inserted, self.failed


def main():
    """Main execution function."""
    # Configuration
    path = "/Users/galex/Downloads/tt_final"
    max_workers = 6  # Adjust based on your system (CPU cores, DB connections)
    
    if not path or not Path(path).exists():
        print(f"Path not found: {path}")
        print("Please update the 'path' variable in this script.")
        return
    
    try:
        from soc_pipeline.config.settings import Config
        cfg = Config()
        dsn = cfg.pg_dsn
        table = "public.mangrove_carbon"
        raster_col = "rast"
        
        print(f"Database: {dsn}")
        print(f"Target table: {table}")
        print(f"Raster column: {raster_col}")
        print(f"Source directory: {path}")
        print(f"Max workers: {max_workers}")
        
        # Create ingester and run
        ingester = TiffIngester(dsn, table, raster_col, max_workers)
        inserted, failed = ingester.ingest_directory(path)
        
        print(f"\n✓ Processing complete!")
        
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
