#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
SERVER_USER_AGENT = "qntfy_ta2"

# Primitives static file directory
D3MSTATICDIR = os.getenv("D3MSTATICDIR", "/static")

# Enable GPU pipelines - "auto" will try to detect, "true" and "false" will force
GPU = os.getenv("GPU", "auto")

# Batch size to apply to primitives where feasible
REMOTE_SENSING_BATCH_SIZE = int(os.getenv("REMOTE_SENSING_BATCH_SIZE", 128))

# Solution serach progress update message interval in seconds
PROGRESS_INTERVAL = float(os.getenv("PROGRESS_INTERVAL", 10.0))

# maximum number of augment columns to support
AUG_MAX_COLS = int(os.getenv("AUG_MAX_COLS", 50))

# maximum number of augment rows to support
AUG_MAX_ROWS = int(os.getenv("AUG_MAX_ROWS", 50000))

# maximum amount of time for hyperparam tuning in seconds
TIME_LIMIT = int(os.getenv("TIME_LIMIT", 600))

# use untuned/internally tuned pipelines (faster) or external tuning (better results)
HYPERPARAMETER_TUNING = os.getenv("HYPERPARAMETER_TUNING", "True") == "True"

# controls parallelism within primitives - defaults to the number of CPUs
N_JOBS = int(os.getenv("N_JOBS", -1))

# enable use of mlp classifier + gradcam visualization
MLP_CLASSIFIER = os.getenv("MLP_CLASSIFIER", "False") == "True"

# whether or not received features for remote sensing are pooled or not
IS_POOLED = os.getenv("POOL_FEATURES", "True") == "True"

COMPUTE_CONFIDENCES = os.getenv("COMPUTE_CONFIDENCES", "False") == "False"
