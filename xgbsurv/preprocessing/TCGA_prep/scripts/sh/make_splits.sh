#!/bin/bash

mkdir -p ./data/splits

python -u scripts/py/rerun_splits.py \
    --data_dir ./data/ \
    --config_path ./