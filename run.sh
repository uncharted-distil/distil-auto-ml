#!/bin/bash

export HYPERPARAMETER_TUNING=False
export COMPUTE_CONFIDENCES=False
export REMOTE_SENSING_BATCH_SIZE=32
export TIME_LIMIT=20
export DEBUG=True

python3 main.py
