import os
import json
import numpy as np
import pandas as pd
from dp_data import Dataset, Domain

def load_domain_config(data_name, root_path='./datasets/preprocessed'):
    domain_path = os.path.join(root_path, f'{data_name}/domain.json')
    config = json.load(open(domain_path))
    return config

def load_df(data_name, filename='data', root_path='./datasets/preprocessed', idxs_path=None):
    data_path = os.path.join(root_path, f'{data_name}')
    df_path = os.path.join(data_path, f'{filename}.csv')
    df = pd.read_csv(df_path)
    if idxs_path is not None:
        idxs_path = os.path.join(data_path, f'idxs/{idxs_path}.npy')
        idxs = np.load(idxs_path)
        df = df.loc[idxs].reset_index(drop=True)
    return df

def get_dataset(data_name, filename='data', root_path='./datasets/preprocessed', idxs_path=None, ignore_numerical=False):
    config = load_domain_config(data_name, root_path=root_path)
    domain = Domain.fromdict(config)
    df = load_df(data_name, filename=filename, root_path=root_path, idxs_path=idxs_path)
    data = Dataset(df, domain)
    if ignore_numerical:
        data = data.project(domain.attrs_cat)
    return data