import random

import pandas as pd
import numpy as np

import lightgbm as lgb

from pprint import pprint

from cv import run_cv_model
from utils import print_step, rmse
from drops import get_drops, save_drops, add_drops
from cache import load_cache, save_in_cache

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 105,
          'max_depth': 8,
          'learning_rate': 0.05,
          'bagging_fraction': 0.95,
          'feature_fraction': 0.8,
          'lambda_l1': 101.3,
          'lambda_l2': 120,
          'min_data_in_leaf': 21,
          'verbosity': -1,
          'data_random_seed': 3,
          'nthread': 4,
          'early_stop': 200,
          'verbose_eval': 100,
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
features = [f for f in features if f not in drops]
print(train[features].shape)
print(test[features].shape)

print('~~~~~~~~~~~~')
leaf_range = [8, 16, 24, 31, 36, 41, 51, 61, 67, 71, 75, 81, 86, 91, 100, 110, 120, 130, 140, 160, 180, 200, 220, 240]
all_results = []
for leaves in leaf_range:
    print('-')
    print_step('Run LGB - Leaves {}'.format(leaves))
    params2 = params.copy()
    random_drops = np.random.choice(features, random.randint(2, 30), replace=False)
    print_step('Randomly dropping {}'.format(', '.join(random_drops)))
    features_c = [f for f in features if f not in random_drops]
    params2['num_leaves'] = leaves
    params2['data_random_seed'] = leaves
    results = run_cv_model(train[features_c], test[features_c], target, runLGB, params2, rmse, 'lgb-{}'.format(leaves))
    all_results.append(results)
import pdb
pdb.set_trace()

pprint(sorted([(leaf_range[i], all_results[i]['final_cv']) for i in range(len(leaf_range))], key = lambda x: x[1]))
rmse(target, np.mean([r['train'] for r in all_results], axis=0))
pd.DataFrame(np.corrcoef([r['train'] for r in all_results]), index = leaf_range, columns = leaf_range)

save_in_cache('leaf_blend3',
			  pd.DataFrame({'leaf_blend3': np.mean([r['train'] for r in all_results], axis=0)}),
			  pd.DataFrame({'leaf_blend3': np.mean([r['test'] for r in all_results], axis=0)}))

submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = np.mean([r['test'] for r in all_results], axis=0)
submission.to_csv('submit/submit_leaf_blend2.csv', index=False)
# CV 3.64285
# Best single: (31, 3.6448112603947362)
