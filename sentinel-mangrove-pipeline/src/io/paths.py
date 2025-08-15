import os
from pathlib import Path

def get_output_path(filename):
    return Path("data/output") / filename

def get_temp_path(filename):
    return Path("data/temp") / filename

def create_directories():
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)