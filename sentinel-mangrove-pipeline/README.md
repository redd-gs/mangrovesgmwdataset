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
├── notebooks/
│   └── exploration.ipynb
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

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/redd-gs/mangrovesgmwdataset/tree/master
   cd sentinel-mangrove-pipeline
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Copy the `.env.example` to `.env` and fill in the required environment variables for database and SentinelHub credentials.

5. **Initialize the Database**
   Use Docker to set up PostgreSQL and PostGIS.
   ```bash
   docker-compose up -d
   ```

6. **Run the Pipeline**
   Execute the pipeline script to start processing.
   ```bash
   ./scripts/run_pipeline.sh
   ```

## Usage Guidelines

- **Ingest Sample Data**: Use the `ingest_sample.sh` script to load sample data into the database.
- **Optimize Database**: Run the `optimize_db.sql` script to apply optimizations to your PostgreSQL database.
- **Explore Data**: Use the Jupyter notebook located in the `notebooks/` directory for exploratory data analysis and visualization.

## Image Processing Enhancements
The project includes various enhancements for image processing, which can be found in the `src/processing/enhancements.py` file. These enhancements aim to improve the quality and speed of image processing tasks.

## Testing
Unit tests are provided in the `tests/` directory. You can run the tests to ensure that the functionality of the pipeline is intact.
```bash
pytest tests/
```
