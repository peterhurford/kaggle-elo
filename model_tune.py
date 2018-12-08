import pandas as pd
import numpy as np

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from drops import get_drops
from cache import load_cache, save_in_cache

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 50,
          'max_depth': 10,
          'learning_rate': 0.05,
          'bagging_fraction': 0.9,
          'feature_fraction': 0.9,
          'lambda_l1': 0.1,
          'lambda_l2': 0,
          'min_data_in_leaf': 30,
          'early_stop': 40,
          'verbose_eval': 2,
          'num_rounds': 10000}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print_step('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    return pred_test_y, pred_test_y2, model.feature_importance()


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = load_cache('data_with_fe')

print_step('Subsetting')
target = train['target']
train_id = train['card_id']
test_id = test['card_id']

features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'target']]
print(train[features].shape)
print(test[features].shape)

drops = get_drops()
features_c = [f for f in features if f not in drops]
print(train[features_c].shape)
print(test[features_c].shape)

print('~~~~~~~~~~~~')
leaf_range = [8, 16, 31, 41, 51, 61, 81, 100, 120, 140, 200]
all_results = []
for leaves in leaf_range:
    print('-')
    print_step('Run LGB - Leaves {}'.format(leaves))
    params2 = params.copy()
    params2['num_leaves'] = leaves
    results = run_cv_model(train[features_c], test[features_c], target, runLGB, params2, rmse, 'lgb-{}'.format(leaves))
    all_results.append(results)
import pdb
pdb.set_trace()
