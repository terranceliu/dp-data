import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from src import cleanup, DataPreprocessor, get_config_from_json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--target', type=str)
    return parser.parse_args()

args = get_args()

data_name = args.dataset
data_dir_raw = f'./datasets/raw/{args.dataset}/'

# Load raw data
df = pd.read_csv(os.path.join(data_dir_raw, 'data.csv'))

# Load attribute dictionary
attrs_dict_path = os.path.join(data_dir_raw, 'attrs.json')
with open(attrs_dict_path, 'r') as f:
    attrs_dict = json.load(f)

# Clean up raw data and corresponding changes to the attribute dictionary
df, attrs_dict = cleanup(args.dataset, df, attrs_dict)

# Create and fit preprocessor
config = get_config_from_json(attrs_dict)
preprocessor = DataPreprocessor(config)
preprocessor.fit(df)

domain = preprocessor.get_domain()
domain[attrs_dict['target']] = domain.pop(attrs_dict['target'])

df_preprocessed = preprocessor.transform(df)

# Save files
data_dir = f'./datasets/preprocessed/{data_name}/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

df_preprocessed.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

domain_path = os.path.join(data_dir, 'domain.json')
with open(domain_path, 'w') as f:
    json.dump(domain, f)

preprocessor_path = os.path.join(data_dir, 'preprocessor.pkl')
with open(preprocessor_path, 'wb') as handle:
    pickle.dump(preprocessor, handle)