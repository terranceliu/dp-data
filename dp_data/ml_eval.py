import argparse
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score
from typing import Tuple, Callable
from dp_data import Domain
from dp_data import get_dataset
from dp_data.ml_models import MODELS, MODEL_PARAMS


def separate_cat_and_num_cols(domain, features):
    # train_cols = [c for c in domain.attrs if c != target]
    train_cols_num = [c for c in features if domain[c] == 1]
    train_cols_cat = [c for c in features if c not in train_cols_num]
    return train_cols_num, train_cols_cat

def get_Xy(domain: Domain, features: list, target, df: pd.DataFrame):
    train_cols_num, train_cols_cat = separate_cat_and_num_cols(domain, features)
    y = df[target].values
    X = None
    if len(train_cols_num) > 0:
        X = df[train_cols_num].values

    if len(train_cols_cat) > 0:
        X_cat = df[train_cols_cat].values
        categories = [np.arange(domain[c]) for c in train_cols_cat]
        enc = OneHotEncoder(categories=categories)
        enc.fit(X_cat)
        X_cat = enc.transform(X_cat).toarray()

        if X is not None:
            X = np.concatenate((X, X_cat), axis=1)
        else:
            X = X_cat

    assert X is not None
    return X, y



def get_evaluate_ml(
        # train_df,
        # test_df,
        df_test,
        config,
        # dataset,
        # data_dir_root,
        # train_test_split_dir,
        # ignore_numerical: bool,
        targets: list,
        models: list,
        grid_search: bool = False,
        verbose: bool=False) -> Callable:
    domain = Domain.fromdict(config)
    # data_train = get_dataset(dataset, root_path=data_dir_root, idxs_path=f'{train_test_split_dir}/train', ignore_numerical=args.ignore_numerical)
    # data_test = get_dataset(dataset, root_path=data_dir_root,
    #                         idxs_path=f'{train_test_split_dir}/test', ignore_numerical=ignore_numerical)
    # domain = data_train.domain
    # target = domain.attrs[-1] if target is None else target

    # targets = ['JWMNP', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)



    models = MODELS.keys() if models is None else models

    def eval_fn(df_train, seed):

        res = []
        for target in targets:
            f1_scoring = 'f1' if domain[target] == 2 else 'f1_macro'
            scorers = {}
            if f1_scoring == 'f1':
                scorers[f1_scoring] = make_scorer(f1_score)
            else:
                scorers[f1_scoring] = make_scorer(f1_score, average='macro')
            scorers['roc'] = make_scorer(roc_auc_score)
            scorers['prc'] = make_scorer(average_precision_score)
            scorers['accuracy'] = make_scorer(accuracy_score)

            X_train, y_train = get_Xy(domain, features=features, target=target, df=df_train)
            X_test, y_test = get_Xy(domain, features=features, target=target, df=df_test)

            if verbose: print(f'Target: {target}:')
            mode = st.mode(y_train).mode[0]
            test_acc_maj = (y_test == mode).mean()
            if verbose: print(f'Majority accuracy: {test_acc_maj}')

            for model_name in models:
                model = MODELS[model_name]
                model.random_state = seed

                import time
                start_time = time.time()

                if grid_search:
                    params = MODEL_PARAMS[model_name]
                    gridsearch = GridSearchCV(model, param_grid=params, cv=args.num_folds, scoring=f1_scoring, verbose=1)
                    gridsearch.fit(X_train, y_train)
                    model = gridsearch.best_estimator_
                    if verbose: print(f'Best parameters: {gridsearch.best_params_}')
                else:
                    model.fit(X_train, y_train)

                if verbose: print(f'Test metrics ({model_name}):')
                for metric_name, scorer in scorers.items():
                    metric_train = scorer(model, X_train, y_train)
                    metric_test = scorer(model, X_test, y_test)
                    if verbose: print(f'Train {metric_name}: {metric_train}')
                    if verbose: print(f'Test {metric_name}: {metric_test}')
                    res.append([target, 'Train', metric_name, metric_train])
                    res.append([target, 'Test', metric_name, metric_test])

                end_time = time.time()
                if verbose: print(f'Total time (s): {end_time - start_time}')
        return pd.DataFrame(res, columns=['target', 'Eval Data', 'Metric', 'Score'])

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

if __name__ == "__main__":
    args = get_args()

    # data_train = get_dataset(args.dataset, root_path=args.data_dir_root, idxs_path=f'{args.train_test_split_dir}/train',
    #                          ignore_numerical=args.ignore_numerical)
    # data_test = get_dataset(args.dataset, root_path=args.data_dir_root,
    #                         idxs_path=f'{args.train_test_split_dir}/test', ignore_numerical=args.ignore_numerical)
    #
    # domain = data_train.domain
    # target = domain.attrs[-1] if args.target is None else args.target
    #
    # evaluate_ml(
    #     train_df=data_train.df,
    #     test_df=data_test.df,
    #     targets=[args.target],
    #     models=args.models,
    #     seed=args.seed,
    #     grid_search=args.grid_search
    # )

    real_train_df, eval_ml = get_evaluate_ml(args.dataset,
                                             data_dir_root=args.data_dir_root,
                                            train_test_split_dir=args.train_test_split_dir,
                                            ignore_numerical=False,
                                            targets=args.targets,
                                            models=['LogisticRegression'])

    eval_ml(real_train_df , 0)