import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score
from typing import Tuple, Callable
from dp_data import Domain
from dp_data import get_dataset
from dp_data.ml_models import MODELS, MODEL_PARAMS


def separate_cat_and_num_cols(domain, features):
    train_cols_num = [c for c in features if domain[c] == 1]
    train_cols_cat = [c for c in features if c not in train_cols_num]
    return train_cols_num, train_cols_cat

def get_Xy(
        domain: Domain, 
        features: list, 
        target, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        scale_real_valued=True
    ):
        
    cols_num, cols_cat = separate_cat_and_num_cols(domain, features)
    y_train = df_train[target].values
    y_test = df_test[target].values

    X_train = None
    X_test = None

    if len(cols_cat) > 0:
        X_cat_train = df_train[cols_cat].values
        X_cat_test = df_test[cols_cat].values
        categories = [np.arange(domain[c]) for c in cols_cat]
        enc = OneHotEncoder(categories=categories)
        enc.fit(X_cat_train)
        X_train = enc.transform(X_cat_train).toarray()
        X_test = enc.transform(X_cat_test).toarray()

    if len(cols_num) > 0:
        X_num_train = df_train[cols_num].values
        X_num_test = df_test[cols_num].values
        if scale_real_valued:
            scaler = StandardScaler()
            scaler.fit(X_num_train)
            X_num_train = scaler.transform(X_num_train)
            X_num_test = scaler.transform(X_num_test)

        if X_train is not None:
            X_train = np.concatenate((X_train, X_num_train), axis=1)
            X_test = np.concatenate((X_test, X_num_test), axis=1)
        else:
            X_train = X_num_train
            X_test = X_num_test

    assert X_train is not None
    assert X_test is not None
    return X_train, y_train, X_test, y_test

def get_scorers():
   scorers = {}
   scorers['f1_binary'] = make_scorer(f1_score, average='binary')
   scorers['f1_micro'] = make_scorer(f1_score, average='micro')
   scorers['f1_macro'] = make_scorer(f1_score, average='macro')
   scorers['f1_weighted'] = make_scorer(f1_score, average='weighted')
   scorers['roc'] = make_scorer(roc_auc_score)
   scorers['prc'] = make_scorer(average_precision_score)
   scorers['accuracy'] = make_scorer(accuracy_score)
   return scorers

def get_evaluate_ml(
        df_test,
        config,
        targets: list,
        models: list,
        grid_search: bool = False,
        rescale=True,
        targets_eval=None,
        ) -> Callable:
    if targets_eval is None:
        targets_eval = targets

    domain = Domain.fromdict(config)
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    models = MODELS.keys() if models is None else models
    scorers = get_scorers()

    def eval_fn(df_train, seed: int=0, verbose: bool=False):
        all_results = []
        for target in targets_eval:
            X_train, y_train, X_test, y_test = get_Xy(domain, 
                                                      features=features, 
                                                      target=target, 
                                                      df_train=df_train, 
                                                      df_test=df_test,
                                                      scale_real_valued=rescale)
            for model_name in models:
                model = MODELS[model_name]
                model.random_state = seed

                print('Fitting...')
                start_time = time.time()

                if grid_search:
                    params = MODEL_PARAMS[model_name]
                    gridsearch = GridSearchCV(model, param_grid=params, cv=5, scoring='f1_macro', verbose=1)
                    gridsearch.fit(X_train, y_train)
                    model = gridsearch.best_estimator_
                    if verbose: print(f'Best parameters: {gridsearch.best_params_}')
                else:
                    model.fit(X_train, y_train)

                end_time = time.time()
                runtime = end_time - start_time

                results = {'target': target, 'model': model_name, 'runtime': runtime}
                for metric_name, scorer in scorers.items():
                    metric_test = scorer(model, X_test, y_test)
                    results[metric_name] = metric_test

                print(results)
                all_results.append(pd.DataFrame([results]))
        return pd.concat(all_results)
    
    return eval_fn




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--targets', nargs='+', type=str, default=None)
    parser.add_argument('--train_test_split_dir', type=str)
    parser.add_argument('--data_dir_root', type=str, default='./datasets/preprocessed')
    parser.add_argument('--ignore_numerical', action='store_true')
    parser.add_argument('--models', nargs='+', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--num_folds', type=int, default=5)
    return parser.parse_args()

#  python dp_data/ml_eval.py --dataset folktables_2018_employment_CA --data_dir_root datasets/preprocessed/folktables/1-Year --train_test_split_dir seed0 --ignore_numerical --models LogisticRegression RandomForest XGBoost

if __name__ == "__main__":
    args = get_args()

    data_train = get_dataset(args.dataset, root_path=args.data_dir_root, idxs_path=f'{args.train_test_split_dir}/train',
                             ignore_numerical=args.ignore_numerical)
    data_test = get_dataset(args.dataset, root_path=args.data_dir_root,
                            idxs_path=f'{args.train_test_split_dir}/test', ignore_numerical=args.ignore_numerical)

    df_train = data_train.df
    df_test = data_test.df

    domain = data_train.domain
    config = domain.config
    targets = [domain.attrs[-1]] if args.targets is None else args.targets
    
    eval_fn = get_evaluate_ml(
        df_test,
        config,
        targets=targets,
        models=args.models,
    )

    df_results = eval_fn(df_train, seed=0)