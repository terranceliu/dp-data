#!/bin/bash

YEAR=2018
STATES='CA NY TX FL PA'
TASKS='income coverage mobility employment travel real multitask'

python run/preprocess_folktables.py --year $YEAR --tasks $TASKS --states $STATES --keep_raw 

SEEDS=$(seq 0 4)

IFS=' ' read -ra STATES <<< "$STATES"
IFS=' ' read -ra TASKS <<< "$TASKS"
for STATE in "${STATES[@]}"
do
    for TASK in "${TASKS[@]}"
    do  
        DATASET=folktables_${YEAR}_${TASK}_${STATE}
        echo $DATASET
        python run/create_splits.py --data_dir_root ./datasets/preprocessed/folktables/1-Year --dataset $DATASET --seeds $SEEDS
    done
done