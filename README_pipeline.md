# Sentinel Mangrove Pipeline

## Overview
The Sentinel Mangrove Pipeline is designed to process satellite imagery from SentinelHub, specifically targeting mangrove ecosystems. This project utilizes PostgreSQL with PostGIS for spatial data management and employs various image processing techniques to enhance the quality of the downloaded images.

## Project Structure
```
sentinel-mangrove-pipeline/
├── README.md
├── .env.example
├── requirements.txt
├── docker/
│   ├── docker-compose.yml
│   └── initdb/
│       └── 001_create_extensions.sql
├── scripts/
│   ├── run_pipeline.sh
│   ├── ingest_sample.sh
│   └── optimize_db.sql
├── src/
│   ├── main.py
│   ├── config/
│   │   └── settings.py
│   ├── db/
│   │   ├── connection.py
│   │   └── queries.py
│   ├── sentinel/
│   │   ├── auth.py
│   │   ├── catalog_search.py
│   │   ├── download.py
│   │   └── evalscripts/
│   │       ├── true_color.js
│   │       └── enhanced_tc.js
│   ├── processing/
│   │   ├── bbox.py
│   │   ├── enhancements.py
│   │   └── tiling.py
│   ├── io/
│   │   ├── paths.py
│   │   └── writer.py
│   └── utils/
│       ├── logging.py
│       └── timing.py
├── data/
│   ├── output/
│   └── temp/
└── tests/
    ├── test_bbox.py
    ├── test_catalog.py
    └── test_enhancements.py
```

## Installation & Usage (Windows PowerShell)

All examples below use Windows PowerShell. If you use Git Bash / WSL, just adapt path separators (`/` instead of `\`).

### 1. Clone the repository
```powershell
git clone https://github.com/redd-gs/mangrovesgmwdataset.git
cd mangrovesgmwdataset/sentinel-mangrove-pipeline
```

### 2. Create & activate the virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked, temporarily allow script execution (current session only):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Environment file
```powershell
Copy-Item .env.example .env
# Then edit .env and fill in credentials (PostgreSQL & SentinelHub)
```

### 5. Start Postgres + PostGIS (Docker)
```powershell
docker compose up -d   # ou: docker-compose up -d
```
To stop:
```powershell
docker compose down
```

### 6. Run the pipeline
Two options:
1. Directly with Python:
```powershell
python .\src\main.py
```
2. Via the existing Bash script (requires Git Bash or WSL):
```powershell
# With Git Bash installed:
bash .\scripts\run_pipeline.sh
```

### 7. (Optional) Recreate / clean the environment
```powershell
deactivate          # if the venv is active (ignored otherwise)
./.venv/Scripts/Activate.ps1  # (if you need to re‑activate before cleaning)
```
Automated cleanup (remove venv + Python caches):
```powershell
./scripts/clear.ps1
```
Only clear output files:
```powershell
./scripts/clearoutputs.ps1 -VerboseList -Recreate
```
(See also the scripts summary table below.)

---

## Scripts & Commands (PowerShell overview)
| Purpose | Script / Action | PowerShell Command |
|----------|-----------------|--------------------|
| Run pipeline | (without Bash) | `python .\src\main.py` |
| Run pipeline (Bash) | `scripts/run_pipeline.sh` | `bash .\scripts\run_pipeline.sh` |
| Sample ingestion (adaptation) | `scripts/ingest_sample.sh` | See adaptation below |
| Optimize DB | `scripts/optimize_db.sql` | `psql -f .\scripts\optimize_db.sql` |
| Clean venv & caches | `scripts/clear.ps1` | `./scripts/clear.ps1` |
| Clear `data/output` | `scripts/clearoutputs.ps1` | `./scripts/clearoutputs.ps1 -Recreate` |
| Run tests | Pytest | `pytest -q` |

### PowerShell adaptation of sample ingestion
The Bash script `ingest_sample.sh` is not directly runnable in pure PowerShell. Equivalent example:
```powershell
# Load environment variables from .env
Get-Content .env | Where-Object { $_ -and ($_ -notmatch '^#') } | ForEach-Object {
   $k,$v = $_ -split '=',2; if($k){ $env:$k = $v }
}

$sample = 'data/sample_data.csv'
psql -h $env:PGHOST -p $env:PGPORT -U $env:PGUSER -d $env:PGDATABASE -c "\\COPY $($env:PGTABLE) FROM '$sample' DELIMITER ',' CSV HEADER;"
```

### Quick DB connectivity check
```powershell
psql "$($env:PGUSER)@$($env:PGHOST)" -d $env:PGDATABASE -c "SELECT NOW();"
```

### Cleaning output files
```powershell
./scripts/clearoutputs.ps1              # simple deletion
./scripts/clearoutputs.ps1 -VerboseList # list deleted files
./scripts/clearoutputs.ps1 -Recreate    # recreate directory after purge
```
Combine options:
```powershell
./scripts/clearoutputs.ps1 -VerboseList -Recreate
```

---

## Running tests (PowerShell)
Unit tests:
```powershell
pytest -q
```
Run a single file:
```powershell
pytest tests\test_bbox.py -q
```
Run a specific test:
```powershell
pytest tests\test_bbox.py::test_bbox_extent -q
```

---

## Quick examples
### Run with temporary parameter overrides
```powershell
$env:MAX_PATCHES = 3
$env:TIME_INTERVAL = '2024-06-01/2024-07-01'
python .\src\main.py
```

### List output paths
```powershell
Get-ChildItem -Recurse .\data\output | Select-Object -First 10
```

### Deactivate the virtual environment
```powershell
deactivate
```

---
## Full reset (local hard reset)
```powershell
# Stop and remove containers + DB volumes
docker compose down -v

# Purge Python environment & caches
./scripts/clear.ps1

# Purge output (and recreate output directory)
./scripts/clearoutputs.ps1 -Recreate

# (Optional) Purge temp
Remove-Item -Recurse -Force data\temp 2>$null
New-Item -ItemType Directory -Path data\temp | Out-Null
```

---
## Notes
* The `.sh` scripts target a Bash environment. On pure Windows, prefer direct Python commands or install Git Bash.
* Ensure Docker Desktop is running before `docker compose up -d`.
* For best performance, avoid running heavy raster operations inside a OneDrive‑synced path if file locks appear.

---
## Quick reference (Linux / WSL Bash)
For quick Linux / WSL reference:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
./scripts/run_pipeline.sh
```


