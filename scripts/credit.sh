#!/bin/bash

# PCA features from original data
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

DATASET=credit
SEEDS=$(seq 0 4)

python run/preprocess.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS