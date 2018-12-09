import random

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
          'verbosity': -1,
          'data_random_seed': 3,
          'nthread': 4,
          'early_stop': 40,
          'verbose_eval': 20,
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

#drops = get_drops()
null_importances, _ = load_cache('null_importances')
drops = list(null_importances[null_importances['gain_score'] <= 0].sort_values('gain_score')['feature'])
features = null_importances[null_importances['gain_score'] >= 0].sort_values('gain_score', ascending=False)['feature'].values 
features_c = [f for f in features if f not in drops]
print(train[features_c].shape)
print(test[features_c].shape)

print('~~~~~~~~~~~~')
print_step('Run Baseline LGB')
results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb-baseline')
best_score = results['final_cv']
total_runs = 60

for run in range(total_runs):    
    if (run + 1) % 30 == 0 and (run + 1) < (total_runs - 20):
        print('-----')
        random.shuffle(features_c)
        random.shuffle(drops)
        random_drop = features_c.pop(0)
        random_add = drops.pop(0)
        drops.insert(2, random_drop)
        features_c.insert(2, random_add)
        print_step('Round {}/{} -- Perturbing... randomly removing {} and randomly adding {}...'.format(run + 1, total_runs, random_drop, random_add))
        print('Shuffling features and drops...')
        print('Current drops are {}'.format(', '.join(drops)))
        print('Features remaining are {}'.format(', '.join(features_c)))
        print('Best score so far is {}'.format(best_score))
    elif (run + 1) % 4 == 0:
        print('-----')
        add = drops.pop(0)
        features_c.append(add)
        print('Round {}/{} -- Trying add {}'.format(run + 1, total_runs, add))
        print('Best score so far is {}'.format(best_score))
        print('Current drops are {}'.format(', '.join(drops)))
    else:
        print('-----')
        drop = features_c.pop(0)
        print('Round {}/{} -- Trying drop {}'.format(run + 1, total_runs, drop))
        print('Best score so far is {}'.format(best_score))
        print('Current drops are {}'.format(', '.join(drops)))
        
    results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb-run-{}'.format(run))
    val_score = results['final_cv']

    if (run + 1) % 30 == 0 and (run + 1) < (total_runs - 20):
        print('...perturbation - reset best score to {}'.format(val_score))
        best_score = val_score
    elif val_score < best_score:
        if (run + 1) % 4 == 0:
            print('...new best! Keeping add of {}.'.format(add))
        else:
            print('...new best! Keeping drop of {}.'.format(drop))
            drops.append(drop)
        best_score = val_score
    else:
        if (run + 1) % 4 == 0:
            drops.append(add)
            features_c = [f for f in features_c if f != add]
        else:
            features_c.append(drop)
    print(train[features_c].shape)

print(train[features_c].shape)
print(test[features_c].shape)

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 50,
          'max_depth': 10,
          'learning_rate': 0.005,
          'bagging_fraction': 0.9,
          'feature_fraction': 0.9,
          'lambda_l1': 0.1,
          'lambda_l2': 0,
          'min_data_in_leaf': 30,
          'verbosity': -1,
          'data_random_seed': 3,
          'nthread': 4,
          'early_stop': 200,
          'verbose_eval': 100,
          'num_rounds': 10000}
print('~~~~~~~~~~~~~~~~~~')
print_step('Run Final LGB')
results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb-final')
print(results['importance'].groupby('feature')['feature', 'importance', 'abs_value'].mean().reset_index().sort_values('abs_value', ascending=False).drop('abs_value', axis=1)) 
import pdb
pdb.set_trace()
