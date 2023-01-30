#!/bin/bash

# Census-income KDD
# https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29

# Download raw data
FILE=./datasets/raw/census/train.csv
if ! test -f "$FILE"; then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
    tar -zxvf census.tar.gz

    HEADER=age,class_of_worker,detailed_industry_recode,detailed_occupation_recode,education,wage_per_hour,enroll_in_edu_inst_last_wk,marital_stat,major_industry_code,major_occupation_code,race,hispanic_origin,sex,member_of_a_labor_union,reason_for_unemployment,full_or_part_time_employment_stat,capital_gains,capital_losses,dividends_from_stocks,tax_filer_stat,region_of_previous_residence,state_of_previous_residence,detailed_household_and_family_stat,detailed_household_summary_in_household,instance_weight,migration_code-change_in_msa,migration_code-change_in_reg,migration_code-move_within_reg,live_in_this_house_1_year_ago,migration_prev_res_in_sunbelt,num_persons_worked_for_employer,family_members_under_18,country_of_birth_father,country_of_birth_mother,country_of_birth_self,citizenship,own_business_or_self_employed,fill_inc_questionnaire_for_veterans_admin,veterans_benefits,weeks_worked_in_year,year,income
    sed -i "1s/^/${HEADER}\n/" census-income.data
    sed -i "1s/^/${HEADER}\n/" census-income.test
    mv census-income.data ./datasets/raw/census/train.csv
    mv census-income.test ./datasets/raw/census/test.csv

    rm -rf census.tar.gz
    rm -rf census-income.names
fi

DATASET=census
SEEDS=$(seq 0 4)

python run/preprocess_train_test.py --dataset $DATASET
python run/create_splits.py --dataset $DATASET --seeds $SEEDS