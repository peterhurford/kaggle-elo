# https://www.kaggle.com/fabiendaniel/hyperparameter-tuning
import warnings

import pandas as pd
import numpy as np

import lightgbm as lgb

from pprint import pprint
from bayes_opt import BayesianOptimization

from cv import run_cv_model
from utils import print_step, rmse
from drops import get_drops
from cache import load_cache, save_in_cache


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


def runBayesOpt(num_leaves, max_depth, bag_fraction, feat_fraction, lambda1, lambda2, min_data):
    print('num_leaves {}, max_depth {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, min_data {}'.format(int(num_leaves), int(max_depth), bag_fraction, feat_fraction, lambda1, lambda2, int(min_data)))
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': int(num_leaves),
              'max_depth': int(max_depth),
              'learning_rate': 0.05,
              'bagging_fraction': bag_fraction,
              'feature_fraction': feat_fraction,
              'lambda_l1': lambda1,
              'lambda_l2': lambda2,
              'min_data_in_leaf': int(min_data),
              'early_stop': 40,
              'verbose_eval': 20,
			  'verbosity': -1,
			  'data_random_seed': 3,
              'num_rounds': 10000}
    results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb')
    val_score = results['final_cv']
    print('score {}: num_leaves {}, max_depth {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, min_data {}'.format(val_score, int(num_leaves), int(max_depth), bag_fraction, feat_fraction, lambda1, lambda2, int(min_data)))
    return -val_score

LGB_BO = BayesianOptimization(runBayesOpt, {
    'num_leaves': (5, 130),
    'max_depth': (4, 10),
    'bag_fraction': (0.1, 1.0),
    'feat_fraction': (0.1, 1.0),
    'lambda1': (0, 10),
    'lambda2': (0, 10),
    'min_data': (10, 400)
})

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=2, n_iter=20, acq='ei', xi=0.0)

pprint(LGB_BO.max)
import pdb
pdb.set_trace()
# {'params': {'bag_fraction': 0.7985187090163345,
#             'feat_fraction': 0.6802921589688393,
#             'lambda1': 9.794769957656374,
#             'lambda2': 9.658390560640429,
#             'max_depth': 9.852726586202104,
#             'min_data': 390.9105383978114,
#             'num_leaves': 79.45735287175796},
#  'target': -3.6652805303686717}

