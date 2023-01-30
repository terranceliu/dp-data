#!/bin/bash

# Census-income KDD
# https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29

DATASET=census
SEEDS=$(seq 0 4)

python run/preprocess_train_test.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS