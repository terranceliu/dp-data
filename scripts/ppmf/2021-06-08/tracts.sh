#!/bin/bash

VERSION=2021-06-08

SEED=0
while getopts s: FLAG
do
case "${FLAG}" in
    s) SEED=${OPTARG};;
esac
done
echo Random seed: $SEED

python run/ppmf/get_random.py --version $VERSION --seeds $SEED
python run/ppmf/get_max_factor.py --version $VERSION 
python run/ppmf/get_quantiles.py --version $VERSION 

FILE_DIR=datasets/preprocessed/ppmf/${VERSION}/geo_sets/tract
FILES=()

FILES+=("${FILE_DIR}/random_${SEED}.txt")
# FILES+=(${FILE_DIR}/max.txt)
# FILES+=(${FILE_DIR}/max_factor_2.txt)
# FILES+=(${FILE_DIR}/max_factor_4.txt)
# FILES+=(${FILE_DIR}/max_factor_8.txt)
# FILES+=(${FILE_DIR}/max_factor_16.txt)
# FILES+=(${FILE_DIR}/mean.txt)
# FILES+=(${FILE_DIR}/quantile_0.25.txt)
# FILES+=(${FILE_DIR}/quantile_0.50.txt)
# FILES+=(${FILE_DIR}/quantile_0.75.txt)

SPLIT_SEEDS=$(seq 0 4)
for FILE in "${FILES[@]}"; do
    echo $FILE
        while read TRACT; do
            echo $TRACT
            python run/ppmf/preprocess_ppmf.py --version $VERSION  --geoid $TRACT

            STATE="${TRACT:0:2}"
            COUNTY="${TRACT:0:5}"
            DATA_DIR_ROOT=./datasets/preprocessed/ppmf/${VERSION}/${STATE}
            for ENTITY in $STATE $COUNTY $TRACT; do
                python run/create_splits.py --data_dir_root ${DATA_DIR_ROOT} --dataset $ENTITY --frac 0.5 --seeds ${SPLIT_SEEDS}
            done
        done <$FILE
done