from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# path of dataset
DATA_DIR        = PROJECT_ROOT / 'data'
RAW_DATA_DIR    = DATA_DIR / 'raw'
CIFAR10_DIR     = RAW_DATA_DIR / 'cifar-10-batches-py'


