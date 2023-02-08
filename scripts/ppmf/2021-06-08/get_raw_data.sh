#!/bin/bash

VERSION=2021-06-08

URL=https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/01-Redistricting_File--PL_94-171/2021-06-08_ppmf_Production_Settings/2021-06-08-ppmf_P.csv

FILE=./datasets/raw/ppmf/${VERSION}/ppmf.csv
if ! test -f "$FILE"; then
    curl -o $FILE $URL
fi

ALL_FIPS=(01 02 04 05 06 08 09 10 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 11 72)

for STATE_ID in "${ALL_FIPS[@]}"
do
    echo $STATE_ID
    python run/ppmf/split_raw_files.py --version $VERSION --stateid $STATE_ID
done