#!/bin/bash

# 1994 Census database - (AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)
# https://archive.ics.uci.edu/ml/datasets/adult",

DATASET=adult
SEEDS=$(seq 0 4)

python run/preprocess_train_test.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS

# python run/preprocess_train_test.py --dataset $DATASET --target $TARGET --prediscretized
# python run/create_splits.py --dataset ${DATASET}_prediscretized --seeds $SEEDS