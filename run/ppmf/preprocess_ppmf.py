import os
import json
import pickle
import argparse
import pandas as pd

from dp_data.ppmf import GeoLocation, select_ppmf_geolocation, get_census_schema_and_data
from dp_data import DataPreprocessor, DataPreprocessingConfig

def get_preprocessor(schema):
    attrs_cat = schema.column_names
    mappings_cat = dict(zip(schema.column_names, schema.column_values))
    config = DataPreprocessingConfig.initialize(attrs_cat=attrs_cat,
                                                mappings_cat = mappings_cat,
                                                )                                      
    return DataPreprocessor(config)

def save_files(save_dir, schema, preprocessor, df_preprocessed, domain):
    schema_path = os.path.join(save_dir, 'schema.pkl')
    with open(schema_path, 'wb') as handle:
        pickle.dump(schema, handle)

    preprocessor_path = os.path.join(save_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as handle:
        pickle.dump(preprocessor, handle)

    csv_path = os.path.join(save_dir, 'data.csv')
    df_preprocessed.to_csv(csv_path, index=False)

    domain_path = os.path.join(save_dir, 'domain.json')
    with open(domain_path, 'w') as f:
        json.dump(domain, f)

parser = argparse.ArgumentParser()
parser.add_argument('--geoid', type=str, default=None, help='tract code')
parser.add_argument('--version', type=str, default='2020-05-27')
args = parser.parse_args()
assert args.version in ('2020-05-27', '2021-06-08'), "invalid PPMF version"

geoids = []
for i in [2, 5, 11, 15]: # state, county, tract, block
    if len(args.geoid) < i:
        break
    geoids.append(args.geoid[:i])

# organize files by state
state_id = geoids[0]
base_dir = f'./datasets/preprocessed/ppmf/{args.version}/{state_id}'

data_dir = f'./datasets/raw/ppmf/{args.version}/state/'
data_path = os.path.join(data_dir, f'ppmf_{state_id}.csv')
ppmf_orig = pd.read_csv(data_path)

# creates files
for geoid in geoids:
    dataset_name = '{}'.format(geoid)

    save_dir = os.path.join(base_dir, f'{dataset_name}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # skip if csv file has already been preprocessed
    if os.path.exists(os.path.join(save_dir, 'data.csv')):
        print(f'{geoid}: files exist')
        continue
        
    geolocation = GeoLocation.parse_geoid(geoid)
    ppmf = select_ppmf_geolocation(ppmf_orig, geolocation)
    schema, ppmf = get_census_schema_and_data(ppmf, ignore_TABBLK=geolocation.type()=='block', version=args.version)
    preprocessor = get_preprocessor(schema)
    df_preprocessed = preprocessor.fit_transform(ppmf)
    domain = preprocessor.get_domain()

    save_files(save_dir, schema, preprocessor, df_preprocessed, domain)
    print(f'{geoid}: {df_preprocessed.shape} {domain}')