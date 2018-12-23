import pandas as pd
import numpy as np

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from drops import get_drops, save_drops, add_drops
from cache import get_data, load_cache, save_in_cache

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 16,
          'max_depth': 8,
          'learning_rate': 0.05,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.8,
          'lambda_l1': 10,
          'lambda_l2': 1,
          'min_data_in_leaf': 40,
          'verbosity': -1,
          'data_random_seed': 3,
          'nthread': 4,
          'early_stop': 40,
          'verbose_eval': 20,
          'num_rounds': 10000}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Prep LGB 1/2')
    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)
    print_step('Prep LGB 2/2')
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
    return pred_test_y, pred_test_y2, None #model.feature_importance()


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Loading base data')
train, test, hist, merch = get_data()
print_step('Munging authorized')
hist['authorized_flag'] = hist['authorized_flag'].map({'Y':1, 'N':0})
print_step('Calculating auth')
auth = hist[hist['authorized_flag'] == 1]
print_step('Calculating auth train')
auth_train = auth.merge(train[['card_id', 'target']], on='card_id')
print_step('Calculating auth test')
auth_test = auth.merge(test[['card_id']], on='card_id')
print_step('Calculating auth target')
auth_target = auth_train.groupby('card_id')[['card_id', 'target']].mean().reset_index()
print_step('Loading vecs')
tr_vec, te_vec = load_cache('char_vects')
results = run_cv_model(tr_vec, te_vec, auth_target['target'], runLGB, params, rmse, 'lgb')
import pdb
pdb.set_trace()
