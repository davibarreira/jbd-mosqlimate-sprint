#!/usr/bin/env python3
"""
Pipeline script to run all notebooks in order to produce submission files.
"""

import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

def run_notebook(notebook_path, timeout=3600):
    """
    Execute a Jupyter notebook and return success status.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
        return True
    except Exception:
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'polars', 'geopandas', 'matplotlib', 'numpy', 'pandas',
        'autogluon', 'epiweeks', 'nbformat', 'nbconvert'
    ]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Missing required package: {package}")
            return False
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'data/1_raw/dengue.csv.gz',
        'data/1_raw/ocean_climate_oscillations.csv.gz',
        'data/1_raw/geodata_uf.geojson'
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required data file: {file_path}")
            return False
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/2_inter',
        'data/3_primary',
        'data/4_model_output',
        'data/5_predictions',
        'train_model/SprintModels'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    if not check_dependencies():
        sys.exit(1)
    if not check_data_files():
        sys.exit(1)
    create_directories()

    pipeline_steps = [
        {'notebook': 'dataprep/1_enso_interpolation.ipynb'},
        {'notebook': 'dataprep/2_geographic_uf.ipynb'},
        {'notebook': 'dataprep/3_join_aggregate_data.ipynb'},
        {'notebook': 'train_model/train.ipynb'},
        {'notebook': 'post_processing/post_processing.ipynb'}
    ]

    start_time = time.time()
    for i, step in enumerate(pipeline_steps, 1):
        print(f"Running step {i}/{len(pipeline_steps)}: {step['notebook']}")
        if not run_notebook(step['notebook']):
            print(f"Step {i} failed: {step['notebook']}")
            sys.exit(1)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Pipeline completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
    print("Submission files should be ready in the predictions folder.")

if __name__ == "__main__":
    main()