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
          'num_leaves': 55,
          'max_depth': 8,
          'learning_rate': 0.01,
          'bagging_fraction': 0.94,
          'feature_fraction': 0.625,
          'lambda_l1': 79.34,
          'lambda_l2': 79.39,
          'min_data_in_leaf': 58,
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

#drops = get_drops()
#print('Current drops are {}'.format(', '.join(drops)))
null_importances, _ = load_cache('null_importances')
drops = list(null_importances[null_importances['gain_score'] <= 0].sort_values('gain_score')['feature'])
print('Current drops are {}'.format(', '.join(drops)))
features_c = [f for f in features if f not in drops]
#features_c = features
print(train[features_c].shape)
print(test[features_c].shape)

print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb')
results['importance']['abs_value'] = abs(results['importance']['importance'])
print(results['importance'].groupby('feature')['feature', 'importance', 'abs_value'].mean().reset_index().sort_values('abs_value', ascending=False).drop('abs_value', axis=1)) 
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lgb', pd.DataFrame({'lgb': results['train']}),
                     pd.DataFrame({'lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = results['test']
submission.to_csv('submit/submit_lgb.csv', index=False)
print_step('Done!')
# [2018-12-11 03:14:22.644050] lgb cv scores : [3.719292275506185, 3.535749965098354, 3.665243866122149, 3.6301183361382807, 3.676172103979167]
# [2018-12-11 03:14:22.644102] lgb mean cv score : 3.645315309368827
# [2018-12-11 03:14:22.644163] lgb std cv score : 0.06173717409674665
# [2018-12-11 03:14:22.653486] lgb final cv score : 3.645837893707764
