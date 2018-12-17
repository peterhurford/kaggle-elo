# https://www.kaggle.com/fabiendaniel/hyperparameter-tuning
import warnings

import pandas as pd
import numpy as np

import lightgbm as lgb

from pprint import pprint
from bayes_opt import BayesianOptimization

from cv import run_cv_model
from utils import print_step, rmse
from drops import get_drops, save_drops, add_drops
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


# oof, submit = load_cache('lgb')
# prior_oof = oof['lgb']


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


def runBayesOpt(num_leaves, bag_fraction, feat_fraction, lambda1, lambda2, min_data):
    print('num_leaves {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, min_data {}'.format(int(num_leaves), bag_fraction, feat_fraction, lambda1, lambda2, int(min_data)))
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': int(num_leaves),
              'max_depth': 11,
              'learning_rate': 0.05,
              'bagging_fraction': bag_fraction,
              'feature_fraction': feat_fraction,
              'lambda_l1': lambda1,
              'lambda_l2': lambda2,
              'min_data_in_leaf': int(min_data),
              'early_stop': 80,
              'verbose_eval': 20,
              'verbosity': -1,
              'data_random_seed': 3,
              'nthread': 4,
              'num_rounds': 10000}
    results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb')
    val_score = results['final_cv']
    # val_score = rmse(target, results['train'] * 0.5 + prior_oof * 0.5)
    print('score {}: num_leaves {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, min_data {}'.format(val_score, int(num_leaves), bag_fraction, feat_fraction, lambda1, lambda2, int(min_data)))
    return -val_score

LGB_BO = BayesianOptimization(runBayesOpt, {
    'num_leaves': (10, 1200),
    'bag_fraction': (0.1, 1.0),
    'feat_fraction': (0.1, 0.9),
    'lambda1': (1, 400),
    'lambda2': (1, 400),
    'min_data': (10, 300)
})

# Bias optimizer toward spots we've found to be good in prior searches
good_spots = [{'bag_fraction': 0.7985187090163345,
               'feat_fraction': 0.6802921589688393,
               'lambda1': 9.794769957656374,
               'lambda2': 9.658390560640429,
               'min_data': 390.9105383978114,
               'num_leaves': 79.45735287175796},
              {'bag_fraction': 0.7315603431126321,
               'feat_fraction': 0.7567557411515091,
               'lambda1': 16.667471886996317,
               'lambda2': 5.119824106597648,
               'min_data': 300.06177710894866,
               'num_leaves': 119.1579612709641},
              {'bag_fraction': 0.1422071191481194,
               'feat_fraction': 0.755712919202501,
               'lambda1': 19.096518651294396,
               'lambda2': 18.784092923070023,
               'min_data': 350.3497092361605,
               'num_leaves': 86.57018015209862},
              {'bag_fraction': 0.879623047331525,
               'feat_fraction': 0.7805371523853327,
               'lambda1': 2.6740934992085927,
               'lambda2': 29.87083549192682,
               'min_data': 200.03214279371076,
               'num_leaves': 80.41956867045907},
              {'bag_fraction': 0.4619456823221404,
               'feat_fraction': 0.7864176698366117,
               'lambda1': 29.503407994650765,
               'lambda2': 28.356333660846193,
               'min_data': 202.10443684992123,
               'num_leaves': 82.96578112428271},
              {'bag_fraction': 0.3184034148477606,
               'feat_fraction': 0.7127473741152386,
               'lambda1': 49.615958886847764,
               'lambda2': 49.86294903319714,
               'min_data': 23.172752369500927,
               'num_leaves': 40.633099999913796},
              {'bag_fraction': 0.8342306889959391,
               'feat_fraction': 0.7196939382136842,
               'lambda1': 58.56969078854843,
               'lambda2': 59.7628238456237,
               'min_data': 49.87730950308235,
               'num_leaves': 20.19216555870919},
              {'bag_fraction': 0.9407993140011,
               'feat_fraction': 0.6252540671424148,
               'lambda1': 79.34759221074741,
               'lambda2': 79.39185979155437,
               'min_data': 57.90790380291228,
               'num_leaves': 55.07234817099164},
              {'bag_fraction': 0.9472662838641918,
               'feat_fraction': 0.797911656390694,
               'lambda1': 101.31794802211229,
               'lambda2': 119.98482321028108,
               'min_data': 21.04059817721408,
               'num_leaves': 105.37921551890115},
			  {'bag_fraction': 0.54,
			   'feat_fraction': 0.73,
			   'lambda1': 197,
			   'lambda2': 3,
			   'min_data': 16,
			   'num_leaves': 26},
			  {'bag_fraction': 0.9,
			   'feat_fraction': 0.77,
			   'lambda1': 183,
			   'lambda2': 155,
			   'min_data': 23,
			   'num_leaves': 499}]
for good_spot in good_spots:
    LGB_BO.probe(params=good_spot, lazy=True)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=8, n_iter=60, acq='ei', xi=0.0)

pprint(sorted([(r['target'], r['params']) for r in LGB_BO.res], reverse=True)[:3])
import pdb
pdb.set_trace()

# Fine tune
LGB_BO.set_bounds(new_bounds={'num_leaves': (20, 700),
                              'lambda1': (30, 300),
                              'lambda2': (30, 300),
                              'min_data': (10, 300)})
LGB_BO.maximize(init_points=0, n_iter=5)
