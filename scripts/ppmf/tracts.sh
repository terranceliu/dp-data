#!/bin/bash

SEED=0
while getopts s: FLAG
do
case "${FLAG}" in
    s) SEED=${OPTARG};;
esac
done
echo Random seed: $SEED

python run/ppmf/get_random.py --seeds $SEED

SPLIT_SEEDS=$(seq 0 4)

while read TRACT; do
    python run/ppmf/preprocess_ppmf.py --geoid $TRACT

    STATE="${TRACT:0:2}"
    COUNTY="${TRACT:0:5}"
    DATA_DIR_ROOT=./datasets/preprocessed/ppmf/${STATE}
    for ENTITY in $STATE $COUNTY $TRACT; do
        python run/create_splits.py --data_dir_root ${DATA_DIR_ROOT} --dataset $ENTITY --frac 0.5 --seeds ${SPLIT_SEEDS}
    done
done <datasets/preprocessed/ppmf/geo_sets/tract/random_${SEED}.txt


    