import os
import copy
import json
import pickle
import shutil
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dp_data import DataPreprocessingConfig, DataPreprocessor

from folktables import BasicProblem
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSTravelTime, ACSMobility

##### Define new task: REAL #####
def real_filter(data):
    df = data
    df = df[df['PWGTP'] >= 1]
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    return df

ACSReal = BasicProblem(
    features=[
        # numerical
        'WKHP', # Usual hours worked per week past 12 months
        'AGEP', # AGE
        'INTP', # Interest, dividends, and net rental income past 12 months
        'JWRIP', # Vehicle occupancy
        'SEMP', # Self-employment income past 12 months
        'WAGP', # Wages or salary income past 12 months
        'POVPIP', # Income-to-poverty ratio recode
        'JWMNP', # Travel time to work
        # categorical
        'PINCP', # Total person's income
        'ESR', # Employment status recode
        'PUBCOV', # Public health coverage
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1P',
    preprocess=real_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def multitask_filter(data):
    """
    Apply filer and binarize the target features: JWMNP PINCP.
    :param data:
    :return:
    """
    df = data
    df = df[df['PWGTP'] >= 1]
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]

    df['PUBCOV'] = (df['PUBCOV'] == 1).astype(int)
    df['ESR'] = (df['ESR'] == 1).astype(int)
    df['JWMNP'] = (df['JWMNP'] > 20).astype(int)
    df['MIG'] = (df['MIG'] == 1).astype(int)
    df['PINCP'] = (df['PINCP'] > 50000).astype(int)

    df : pd.DataFrame
    df = df.rename(columns={'JWMNP': 'JWMNP_bin'})
    return df

ACSMultitask = BasicProblem(
    features=[
        # categorical features
        'COW', # Class of worker
        'SCHL', # Educational attainment
        'MAR', # Marital status
        'RELP', # Relationship (to reference person)
        'SEX', #  Male or Female
        'RAC1P', # Race
        'WAOB', # World area of birth
        'FOCCP', # Occupation
        'DIS', # Disability
        'ESP', # Employment status of parents
        'CIT', # Citizenship status
        'JWTR', # Means of transportation to work
        'MIL', # Served September 2001 or later
        'ANC', # Ancestry
        'NATIVITY', # Nativity
        'DEAR', # Hearing difficulty
        'DEYE', # Vision difficulty
        'DREM', # Cognitive difficulty
        'GCL', # Grandparents living with grandchildren
        'FER', # Gave birth to child within the past 12 months
        'PUMA', # Public use microdata area cod
        'POWPUMA', # Place of work PUMA
        'OCC', # Occupation
        # numerical features
        'WKHP', # Usual hours worked per week past 12 months
        'AGEP', # AGE
        'INTP', # Interest, dividends, and net rental income past 12 months
        'JWRIP', # Vehicle occupancy
        'SEMP', # Self-employment income past 12 months
        'WAGP', # Wages or salary income past 12 months
        'POVPIP', # Income-to-poverty ratio recode
        # target variables
        'JWMNP', # Travel time to work. Is it greater than 20?
        'PINCP', # Total person's income. Is it greater than 50K?
        'ESR', # Employment status recode
        'MIG', # Mobility status (lived here 1 year ago)
        'PUBCOV', # Public health coverage
    ],
    target='PINCP',
    target_transform=None,
    group='RAC1P',
    preprocess=multitask_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)
# we rename JWMNP so that it doesn't automatically get mapped to 1 as a numerical column (ALL_ATTRS_NUM)
# the preprocessing script ignores Task.target, so the same isn't required for PINCP
ACSMultitask.features.remove('JWMNP')
ACSMultitask.features.append('JWMNP_bin')


##### Data info #####
RAW_DATA_DIR = './datasets/raw/folktables'

ALL_STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

ALL_ATTRS_NUM = ['AGEP', 'FINCP', 'GRNTP', 'GRPIP', 'HINCP', 'INSP', 'INTP',
                  'JWMNP', 'JWRIP', 'MARHYP', 'MHP', 'MRGP', 'NOC', 'NPF', 'NRC',
                  'OCPIP', 'OIP', 'PAP', 'PERNP', 'PINCP', 'POVPIP', 'PWGTP', 'RETP',
                  'RMSP', 'RNTP', 'SEMP', 'SMOCP', 'SMP', 'SSIP', 'SSP', 'TAXAMT',
                  'VALP', 'WAGP', 'WATP', 'WGTP1', 'WKHP', 'YOEP']

COLS_DEL = ['ST']
COLS_STATE_SPECIFIC = ['PUMA', 'POWPUMA']

# When adding a new task, you must verify whether continuous attributes are included in ALL_CONT_ATTRS
ACSTask = {
    'income': ACSIncome,
    'coverage': ACSPublicCoverage,
    'mobility': ACSMobility,
    'employment': ACSEmployment,
    'travel': ACSTravelTime,
    'real': ACSReal,
    'multitask': ACSMultitask
}

# get list of catgorical and numerical attributes
def split_cat_num(attrs, target_attr=None):
    all_attrs_num = ALL_ATTRS_NUM.copy()
    if target_attr is not None and target_attr in all_attrs_num: # target attr is always binary
        all_attrs_num.remove(target_attr)
    attrs_cat = set(attrs) - set(all_attrs_num)
    attrs_num = set(attrs).intersection(all_attrs_num)
    return list(attrs_cat), list(attrs_num)

def get_acs_raw(task, state, year='2018', horizon='1-Year', remove_raw_files=False, return_attrs=False):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person',
                                root_dir=RAW_DATA_DIR)
    acs_data = data_source.get_data(states=[state], download=True)
    features, target, group = ACSTask[task].df_to_numpy(acs_data)

    all_attrs = ACSTask[task].features.copy()
    df = pd.DataFrame(features, columns=all_attrs)

    target_attr = ACSTask[task].target
    all_attrs.append(target_attr)
    df[target_attr] = target.astype(features.dtype)

    if remove_raw_files:
        shutil.rmtree(data_source._root_dir)

    attr_cat, attr_num = split_cat_num(all_attrs, target_attr=target_attr)
    if return_attrs:
        return df, ACSTask[task].target, (attr_cat, attr_num)
    return df, ACSTask[task].target

def get_preprocessor_mappings(task):
    dict_cat, dict_num = {}, {}
    for state in tqdm(ALL_STATES):
        df, _, (attr_cat, attr_num) = get_acs_raw(task, state, return_attrs=True)

        for attr in attr_cat:
            unique_attrs = set(df[attr].unique().astype(int))
            if attr in dict_cat.keys():
                dict_cat[attr] = dict_cat[attr].union(unique_attrs)
            else:
                dict_cat[attr] = unique_attrs

        for attr in attr_num:
            min_val, max_val = df[attr].min(), df[attr].max()
            if attr in dict_num.keys():
                curr_min_val, curr_max_val = dict_num[attr]
                if min_val < curr_min_val:
                    dict_num[attr][0] = min_val
                if max_val > curr_max_val:
                    dict_num[attr][1] = max_val
            else:
                dict_num[attr] = [min_val, max_val]

    for key, val in dict_cat.items():
        dict_cat[key] = list(val)

    return dict_cat, dict_num


def preprocess_acs(task, state, year='2018', horizon='1-Year'):
    df, target = get_acs_raw(task, state)
    
    mappings_dir = os.path.join(RAW_DATA_DIR, year, horizon, 'preprocessor_mappings')
    if not os.path.exists(mappings_dir):
        os.makedirs(mappings_dir)
    mappings_path = os.path.join(mappings_dir, '{}.pkl'.format(task))
    if os.path.exists(mappings_path):
        with open(mappings_path, 'rb') as handle:
            dict_cat, dict_num = pickle.load(handle)
    else:
        dict_cat, dict_num = get_preprocessor_mappings(task)
        with open(mappings_path, 'wb') as handle:
            pickle.dump((dict_cat, dict_num), handle)

    for attr in COLS_DEL:
        dict_cat.pop(attr, None)
        dict_num.pop(attr, None)

    # use state instead of national values to create mappings
    for attr in COLS_STATE_SPECIFIC:
        if attr in dict_cat.keys():
            dict_cat[attr] = list(np.unique(df[attr].unique()))
        elif attr in dict_num.keys():
            min_val, max_val = df[attr].min(), df[attr].max()
            dict_num[attr] = (min_val, max_val)
    # remove categorical attrs that take on a single value
    attrs_remove = [attr for attr, mapping in dict_cat.items() if len(mapping) == 1]
    for attr in attrs_remove:
        del dict_cat[attr]

    config = DataPreprocessingConfig.initialize(attrs_cat=list(dict_cat.keys()), 
                                                attrs_num=list(dict_num.keys()),
                                                mappings_cat=copy.deepcopy(dict_cat),
                                                mappings_num=copy.deepcopy(dict_num))
    preprocessor = DataPreprocessor(config)
                                                
    df_preprocessed = preprocessor.fit_transform(df)
    domain = preprocessor.get_domain()
    domain[target] = domain.pop(target)

    return df_preprocessed, domain, preprocessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--states', nargs='+')
    parser.add_argument('--year', type=str, default='2018')
    parser.add_argument('--horizon', type=str, default='1-Year')
    parser.add_argument('--keep_raw', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print(args)
    for task, state in itertools.product(args.tasks, args.states):
        print(task, state)
        df_preprocessed, domain, preprocessor = preprocess_acs(task, state, year=args.year, horizon=args.horizon)

        data_dir = f'./datasets/preprocessed/folktables/{args.horizon}/folktables_{args.year}_{task}_{state}'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        df_preprocessed.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

        domain_path = os.path.join(data_dir, 'domain.json')
        with open(domain_path, 'w') as f:
            json.dump(domain, f)

        preprocessor_path = os.path.join(data_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'wb') as handle:
            pickle.dump(preprocessor, handle)

    mappings_dir = os.path.join(RAW_DATA_DIR, args.year, args.horizon, 'preprocessor_mappings')
    shutil.rmtree(mappings_dir)
    if not args.keep_raw:
        shutil.rmtree(RAW_DATA_DIR)