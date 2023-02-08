import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import pdb

STATE_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '12', '13', 
             '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 
             '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 
             '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', 
             '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', 
             '11', '72']

parser = argparse.ArgumentParser()
parser.add_argument('--state_ids', nargs='+', type=str, default=STATE_IDS, help='selects random entity within state')
parser.add_argument('--seeds', nargs='+', type=int, default=[0])
parser.add_argument('--version', type=str, default='2020-05-27')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
assert args.version in ('2020-05-27', '2021-06-08'), "invalid PPMF version"

random_geoids_dict, prng_dict = {}, {}
for seed in args.seeds:
    random_geoids_dict[seed] = {'county': [], 'tract': [], 'block': []}
    prng_dict[seed] = np.random.RandomState(seed)

save_dir_base = f'./datasets/preprocessed/ppmf/{args.version}/geo_sets/'
# If all the files for all seeds exist, exit.
# Note that this doesn't check whether each file contains geoids for all states passed into the arguments
check = True
for seed in args.seeds:
    for entity in random_geoids_dict[seed].keys():
        save_dir = os.path.join(save_dir_base, entity)
        save_path = os.path.join(save_dir, f'random_{seed}.txt')
        check &= os.path.exists(save_path)
if not args.overwrite and check:
    exit()

data_dir = f'./datasets/raw/ppmf/{args.version}/state/'
for state_id in tqdm(args.state_ids):
    data_path = os.path.join(data_dir, f'ppmf_{state_id}.csv')
    ppmf = pd.read_csv(data_path, usecols=['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK'])

    ppmf['geoid_county'] = ppmf['TABBLKST'].apply(lambda x: str(x).zfill(2)) + ppmf['TABBLKCOU'].apply(lambda x: str(x).zfill(3))
    ppmf['geoid_tract'] = ppmf['geoid_county'] + ppmf['TABTRACTCE'].apply(lambda x: str(x).zfill(6))
    ppmf['geoid_block'] = ppmf['geoid_tract'] + ppmf['TABBLK'].apply(lambda x: str(x).zfill(4))

    for seed in args.seeds:
        for entity in random_geoids_dict[seed].keys():
            geoids = ppmf[f'geoid_{entity}'].unique()
            geoid = prng_dict[seed].choice(geoids)
            random_geoids_dict[seed][entity].append(geoid)

for seed in args.seeds:
    for entity, geoids in random_geoids_dict[seed].items():
        save_dir = os.path.join(save_dir_base, entity)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'random_{seed}.txt')
        with open(save_path, "w") as file:
            file.write('\n'.join(geoids))