import os
import json
import numpy as np
import pandas as pd
from dp_data import Dataset, Domain

def get_domain(data_name, root_path='./datasets/'):
    domain_path = os.path.join(root_path, f'{data_name}/domain.json')
    config = json.load(open(domain_path))
    domain = Domain(config.keys(), config.values())
    return domain

def get_data(data_name, filename='data', root_path='./datasets/', idxs_path=None):
    data_path = os.path.join(root_path, f'{data_name}')
    df_path = os.path.join(data_path, f'{filename}.csv')
    df = pd.read_csv(df_path)
    if idxs_path is not None:
        idxs_path = os.path.join(data_path, f'idxs/{idxs_path}.npy')
        idxs = np.load(idxs_path)
        df = df.loc[idxs].reset_index(drop=True)
    return df

def get_dataset(data_name, filename='data', root_path='./datasets/', idxs_path=None):
    domain = get_domain(data_name, root_path=root_path)
    df = get_data(data_name, filename=filename, root_path=root_path, idxs_path=idxs_path)
    return Dataset(df, domain)