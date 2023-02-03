import os
import argparse
import pandas as pd
from tqdm import tqdm

STATE_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '12', '13', 
             '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 
             '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 
             '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', 
             '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', 
             '11']
ENTITIES = ['county', 'tract', 'block']

def idxquantile(s, q=0.5, *args, **kwargs):
    qv = s.quantile(q, *args, **kwargs)
    return (s.sort_values()[::-1] <= qv).idxmax()

parser = argparse.ArgumentParser()
parser.add_argument('--state_ids', nargs='+', type=str, default=STATE_IDS)
parser.add_argument('--quantiles', nargs='+', type=float, default=[0.25, 0.5, 0.75])
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

random_geoids_dict = {}
for q in args.quantiles:
    random_geoids_dict[q] = {'county': [], 'tract': [], 'block': []}

# If all the files for all seeds exist, exit.
# Note that this doesn't check whether each file contains geoids for all states passed into the arguments
check = True
for q in args.quantiles:
    for entity in random_geoids_dict[q].keys():
        save_dir = f'./datasets/preprocessed/ppmf/geo_sets/{entity}'
        save_path = os.path.join(save_dir, f'quantile_{q:.2f}.txt')
        check &= os.path.exists(save_path)
if not args.overwrite and check:
    exit()

for state_id in tqdm(args.state_ids):
    data_dir = './datasets/raw/ppmf/state/'
    data_path = os.path.join(data_dir, f'ppmf_{state_id}.csv')
    ppmf = pd.read_csv(data_path, usecols=['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK'])

    ppmf['geoid_county'] = ppmf['TABBLKST'].apply(lambda x: str(x).zfill(2)) + ppmf['TABBLKCOU'].apply(lambda x: str(x).zfill(3))
    ppmf['geoid_tract'] = ppmf['geoid_county'] + ppmf['TABTRACTCE'].apply(lambda x: str(x).zfill(6))
    ppmf['geoid_block'] = ppmf['geoid_tract'] + ppmf['TABBLK'].apply(lambda x: str(x).zfill(4))

    for q in args.quantiles:
        for entity in random_geoids_dict[q].keys():
            x = ppmf.groupby(f'geoid_{entity}').size()
            geoid = idxquantile(x, q=q)
            random_geoids_dict[q][entity].append(geoid)

for q in args.quantiles:
    for entity, geoids in random_geoids_dict[q].items():
        save_dir = f'./datasets/preprocessed/ppmf/geo_sets/{entity}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'quantile_{q:.2f}.txt')
        with open(save_path, "w") as file:
            file.write('\n'.join(geoids))