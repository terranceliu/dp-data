import os
import argparse
import pandas as pd
from tqdm import tqdm

"""
saves
1) the largest county/tract/block
2) closest to the mean
3) closest to max_size / factor for factor in args.factor
"""

STATE_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '12', '13', 
             '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 
             '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 
             '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', 
             '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', 
             '11']
ENTITIES = ['county', 'tract', 'block']

def idxclosest(s, val):
    return (s.sort_values()[::-1] <= val).idxmax()

parser = argparse.ArgumentParser()
parser.add_argument('--state_ids', nargs='+', type=str, default=STATE_IDS)
parser.add_argument('--factors', nargs='+', type=float, default=[2, 4, 8, 16])
parser.add_argument('--version', type=str, default='2020-05-27')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
assert args.version in ('2020-05-27', '2021-06-08'), "invalid PPMF version"

geoids_dict = {}
for key in ['max', 'mean'] + args.factors:
    geoids_dict[key] = {}
    for entity in ENTITIES:
        geoids_dict[key][entity] = []

save_dir_base = f'./datasets/preprocessed/ppmf/{args.version}/geo_sets/'
# If all the files for all seeds exist, exit.
# Note that this doesn't check whether each file contains geoids for all states passed into the arguments
check = True
for entity in ENTITIES:
    save_dir = os.path.join(save_dir_base, entity)
    save_path = os.path.join(save_dir, f'mean.txt')
    check &= os.path.exists(save_path)
    for factor in args.factors:
        save_path = os.path.join(save_dir, f'max_factor_{factor}.txt')
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

    for entity in ENTITIES:
        x = ppmf.groupby(f'geoid_{entity}').size()
        mean_size, max_size = x.mean(), x.max()
        geoids_dict['mean'][entity].append(idxclosest(x, mean_size))
        geoids_dict['max'][entity].append(idxclosest(x, max_size))
        for factor in args.factors:
            val = max_size / factor
            geoid = idxclosest(x, val)
            geoids_dict[factor][entity].append(geoid)

for entity in ENTITIES:
    save_dir = os.path.join(save_dir_base, entity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    geoids = geoids_dict['mean'][entity]
    save_path = os.path.join(save_dir, f'mean.txt')
    with open(save_path, "w") as file:
        file.write('\n'.join(geoids))

    geoids = geoids_dict['max'][entity]
    save_path = os.path.join(save_dir, f'max.txt')
    with open(save_path, "w") as file:
        file.write('\n'.join(geoids))

    for factor in args.factors:
        geoids = geoids_dict[factor][entity]
        save_path = os.path.join(save_dir, f'max_factor_{factor}.txt')
        with open(save_path, "w") as file:
            file.write('\n'.join(geoids))