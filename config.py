import os

DB_LOCATION = os.getenv("DB_URI", "test.db")

# Debug flag to output more verbose logging
# - defaults to False
DEBUG = os.getenv("DEBUG", False)

# Configurable output directory for saving machine learning model pickles
# - defaults to ../output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Port to make worker service available on
PORT = os.getenv("PORT", "45042")

# Configurable filename for output logs
LOG_FILENAME = os.getenv("LOG_FILENAME", "distil-auto-ml.log")

# User agent to supply to TA3 Systems
SERVER_USER_AGENT='qntfy_ta2'

# Primitives static file directory
D3MSTATICDIR = os.getenv("D3MSTATICDIR", '/static')

# Enable GPU pipelines
GPU = os.getenv("GPU", False)
