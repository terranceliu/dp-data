#!/bin/bash

# Diagnosis of cervical cancer
# Created a single target variable for whether person tests positive with any four of the methods
# https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

DATASET=cervical
SEEDS=$(seq 0 4)

python run/preprocess.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS