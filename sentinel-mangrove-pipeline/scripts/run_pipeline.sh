#!/bin/bash

# Activate the Python virtual environment
source ../venv/bin/activate

# Set environment variables from the .env file
export $(cat ../.env.example | xargs)

# Run the main pipeline script
python ../src/main.py

# Deactivate the virtual environment
deactivate