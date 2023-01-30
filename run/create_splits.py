import os
import numpy as np
import argparse
from src import get_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir_root', type=str, default='./datasets')
    parser.add_argument('--frac', type=float, default=0.8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    return parser.parse_args()

def split_data(df, frac=0.8, seed=0):
    idxs_train = df.sample(frac=frac, random_state=seed).index
    idxs_test = df.drop(idxs_train).index
    return idxs_train.values, idxs_test.values

args = get_args()

for seed in args.seeds:
    data_dir = os.path.join(args.data_dir_root, f'{args.dataset}/idxs/seed{seed}/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df = get_dataset(args.dataset, root_path=args.data_dir_root).df
    idxs_train, idxs_test = split_data(df, frac=args.frac, seed=seed)

    np.save(os.path.join(data_dir, 'train.npy'), idxs_train)
    np.save(os.path.join(data_dir, 'test.npy'), idxs_test)