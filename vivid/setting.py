import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

FEATURE_DIR = os.path.join(DATA_DIR, 'feature')
os.makedirs(FEATURE_DIR, exist_ok=True)

PROCESSED_ROOT = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_ROOT, exist_ok=True)

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

HISTORY_DIR = os.path.join(DATA_DIR, 'history')
os.makedirs(HISTORY_DIR, exist_ok=True)

VIS_DIR = os.path.join(DATA_DIR, 'visualize')
os.makedirs(VIS_DIR, exist_ok=True)
