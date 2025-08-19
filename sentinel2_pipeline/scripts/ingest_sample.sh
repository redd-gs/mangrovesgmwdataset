#!/bin/bash

# Ingest sample data into the PostgreSQL database

# Load environment variables from .env file
set -a
source .env
set +a

# Database connection parameters
DB_HOST=${PGHOST:-localhost}
DB_PORT=${PGPORT:-5432}
DB_NAME=${PGDATABASE:-global_mangrove_dataset}
DB_USER=${PGUSER:-postgres}
DB_PASSWORD=${PGPASSWORD:-mangrovesondra}

# Sample data file path
SAMPLE_DATA_PATH="data/sample_data.csv"

# Ingest sample data into the database
echo "Ingesting sample data from $SAMPLE_DATA_PATH into $DB_NAME..."

# Execute the SQL command to ingest data
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "\COPY gmw_2016_v2 FROM '$SAMPLE_DATA_PATH' DELIMITER ',' CSV HEADER;"

if [ $? -eq 0 ]; then
    echo "Sample data ingested successfully."
else
    echo "Error ingesting sample data."
fi

SELECT Find_SRID(PG_catalog.current_schema(), 'gmw_2016_v2', 'geom');
SELECT ST_SRID(geom), ST_Extent(geom) FROM public.gmw_2016_v2 LIMIT 5;

BBox([2.293,48.857,2.298,48.860], crs=CRS.WGS84)

from core.context import get_sh_config
cfg = get_sh_config()
print(cfg.sh_client_id, cfg.sh_client_secret[:4]+"***")
print(img.min(), img.max(), img.mean())
print("[DEBUG] SH_CLIENT_ID:", settings().SH_CLIENT_ID)