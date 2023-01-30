#!/bin/bash

# Marketing campaigns of a Portuguese banking institution (bank-additional)
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

DATASET=bank
SEEDS=$(seq 0 4)

python run/preprocess.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS