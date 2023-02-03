import argparse
import numpy as np
from scipy import stats as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score

from dp_data import get_dataset
from dp_data.ml_models import MODELS, MODEL_PARAMS

def get_train_cols(domain) :
   train_cols = [c for c in domain.attrs if c != target]
   train_cols_num = [c for c in train_cols if domain[c] == 1]
   train_cols_cat = [c for c in train_cols if c not in train_cols_num]
   return train_cols, train_cols_num, train_cols_cat


def get_train_test(domain, df_train, df_test, target):
   _, train_cols_num, train_cols_cat = get_train_cols(domain)
   y_train, y_test = df_train[target].values, df_test[target].values


   X_train, X_test = df_train[train_cols_num].values, df_test[train_cols_num].values
   X_train_cat, X_test_cat = df_train[train_cols_cat].values, df_test[train_cols_cat].values


   if len(train_cols_cat) > 0:
       categories = [np.arange(domain[c]) for c in train_cols_cat]
       enc = OneHotEncoder(categories=categories)
      
       enc.fit(X_train_cat)
       X_train_cat = enc.transform(X_train_cat).toarray()
       X_test_cat = enc.transform(X_test_cat).toarray()

       X_train = np.concatenate((X_train, X_train_cat), axis=1)
       X_test = np.concatenate((X_test, X_test_cat), axis=1)


   return X_train, y_train, X_test, y_test


def get_args():
   parser = argparse.ArgumentParser()
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str)
   parser.add_argument('--target', type=str, default=None)
   parser.add_argument('--train_test_split_dir', type=str)
   parser.add_argument('--data_dir_root', type=str, default='./datasets/preprocessed')
   parser.add_argument('--ignore_numerical', action='store_true')
   parser.add_argument('--models', nargs='+', type=str, default=None)
   parser.add_argument('--seed', type=int, default=0)
   parser.add_argument('--grid_search', action='store_true')
   parser.add_argument('--num_folds', type=int, default=5)
   return parser.parse_args()

args = get_args()

data_train = get_dataset(args.dataset, root_path=args.data_dir_root, 
                         idxs_path=f'{args.train_test_split_dir}/train', ignore_numerical=args.ignore_numerical)
data_test = get_dataset(args.dataset, root_path=args.data_dir_root, 
                        idxs_path=f'{args.train_test_split_dir}/test', ignore_numerical=args.ignore_numerical)

domain = data_train.domain
target = domain.attrs[-1] if args.target is None else args.target

f1_scoring = 'f1' if domain[target] == 2 else 'f1_macro'

scorers = {}
if f1_scoring == 'f1':
   scorers[f1_scoring] = make_scorer(f1_score)
else:
   scorers[f1_scoring] = make_scorer(f1_score, average='macro')
scorers['roc'] = make_scorer(roc_auc_score)
scorers['prc'] = make_scorer(average_precision_score)
scorers['accuracy'] = make_scorer(accuracy_score)

X_train, y_train, X_test, y_test = get_train_test(domain, data_train.df, data_test.df, target)

mode = st.mode(y_train, keepdims=True).mode[0]
test_acc_maj = (y_test == mode).mean()
print(f'Majority accuracy: {test_acc_maj}')

models = MODELS.keys() if args.models is None else args.models
for model_name in models:
   model = MODELS[model_name]
   model.random_state = args.seed

   import time
   start_time = time.time()

   if args.grid_search:
      params = MODEL_PARAMS[model_name]
      gridsearch = GridSearchCV(model, param_grid=params, cv=args.num_folds, scoring=f1_scoring, verbose=1)
      gridsearch.fit(X_train, y_train)
      model = gridsearch.best_estimator_
      print(f'Best parameters: {gridsearch.best_params_}')
   else:
      model.fit(X_train, y_train)

   print(f'Test metrics ({model_name}):')
   for metric_name, scorer in scorers.items():
       metric_train = scorer(model, X_train, y_train)
       metric_test = scorer(model, X_test, y_test)
       print(f'{metric_name}: {metric_test}')

   end_time = time.time()
   print(f'Total time (s): {end_time - start_time}')