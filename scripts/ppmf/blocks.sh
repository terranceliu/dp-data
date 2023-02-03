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
python run/ppmf/get_max_factor.py
python run/ppmf/get_quantiles.py

FILE_DIR=datasets/preprocessed/ppmf/geo_sets/block
FILES=()
FILES+=(${FILE_DIR}/random_${SEED}.txt)
FILES+=(${FILE_DIR}/max.txt)
FILES+=(${FILE_DIR}/max_factor_2.txt)
FILES+=(${FILE_DIR}/max_factor_4.txt)
FILES+=(${FILE_DIR}/max_factor_8.txt)
FILES+=(${FILE_DIR}/max_factor_16.txt)
FILES+=(${FILE_DIR}/mean.txt)
FILES+=(${FILE_DIR}/quantile_0.25.txt)
FILES+=(${FILE_DIR}/quantile_0.50.txt)
FILES+=(${FILE_DIR}/quantile_0.75.txt)

SPLIT_SEEDS=$(seq 0 4)
for FILE in "${FILES[@]}"; do
    echo $FILE
    while read BLOCK; do
        echo $BLOCK
        python run/ppmf/preprocess_ppmf.py --geoid $BLOCK

        STATE="${BLOCK:0:2}"
        COUNTY="${BLOCK:0:5}"
        TRACT="${BLOCK:0:11}"
        DATA_DIR_ROOT=./datasets/preprocessed/ppmf/${STATE}
        for ENTITY in $STATE $COUNTY $TRACT $BLOCK; do
            python run/create_splits.py --data_dir_root ${DATA_DIR_ROOT} --dataset $ENTITY --frac 0.5 --seeds ${SPLIT_SEEDS}
        done
    done <$FILE
done