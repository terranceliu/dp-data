#!/bin/bash

YEAR=2018
STATES='CA NY TX FL PA'
TASKS='income coverage mobility employment travel real multitask'

python run/preprocess_folktables.py --year 2018 --tasks real --states 'CA' --keep_raw --remove_outliers

python run/create_splits.py --data_dir_root ./datasets/preprocessed/folktables/1-Year --dataset folktables_2018_real_CA_outliers --seeds 0 1 2
