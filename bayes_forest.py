# https://www.kaggle.com/fabiendaniel/hyperparameter-tuning
import warnings

import pandas as pd
import numpy as np

import lightgbm as lgb

from pprint import pprint
from bayes_opt import BayesianOptimization, UtilityFunction

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
              'early_stop': 40,
              'verbose_eval': 20,
              'verbosity': -1,
              'data_random_seed': 3,
              'nthread': 4,
              'num_rounds': 10000}
    return run_cv_model(train[features_c], test[features_c], target, runLGB, params, rmse, 'lgb')

LGB_BO = BayesianOptimization(runBayesOpt, {
    'num_leaves': (10, 1200),
    'bag_fraction': (0.1, 1.0),
    'feat_fraction': (0.1, 0.9),
    'lambda1': (1, 400),
    'lambda2': (1, 400),
    'min_data': (10, 300)
})
print_step('Baseline')
results = runBayesOpt(num_leaves=105,
                     bag_fraction=0.95,
                     feat_fraction=0.8,
                     lambda1=101.3,
                     lambda2=120,
                     min_data=21)
print_step('{}: {}'.format('baseline', results['final_cv']))
best_score = results['final_cv']
current_train_oofs = [results['train']]
current_test_oofs = [results['test']]
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    abandon = False
    i = 2
    while i < 21 and not abandon:
        print('-')
        print('-')
        print('-')
        print_step('{}/20'.format(i))
        i += 1
        LGB_BO = BayesianOptimization(runBayesOpt, {
            'num_leaves': (10, 1200),
            'bag_fraction': (0.1, 1.0),
            'feat_fraction': (0.1, 0.9),
            'lambda1': (1, 400),
            'lambda2': (1, 400),
            'min_data': (10, 300)
        })
        utility = UtilityFunction(kind='ucb', kappa=2.5, xi=0.0)
        done = False
        tries = 0
        while not done:
            next_point_to_probe = LGB_BO.suggest(utility)
            results = runBayesOpt(**next_point_to_probe)
            weight = 1 / i
            score = rmse(target, (1 - weight) * np.mean(np.array(current_train_oofs), axis=0) + weight * results['train'])
            if score < best_score:
                done = True
                current_train_oofs.append(results['train'])
                current_test_oofs.append(results['test'])
                best_score = score
            else:
                tries += 1
                if tries > 20:
                    print_step('20 tries exceeded... abandoning')
                    done = True
                    abandon = True
            LGB_BO.register(params=next_point_to_probe, target=-score)
            print_step('{}: Local {} - Global {} - Best {}'.format(next_point_to_probe, results['final_cv'], score, best_score))
            print('-')

import pdb
pdb.set_trace()

rmse(target, np.mean([o for o in current_train_oofs], axis=0))
pd.DataFrame(np.corrcoef(current_train_oofs))
[rmse(target, o) for o in current_train_oofs]
[rmse(target, np.mean(current_train_oofs[:i], axis=0)) for i in np.array(range(len(current_train_oofs)))[1:]]

submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = np.mean(current_test_oofs, axis=0)
submission.to_csv('submit/submit_bayes_blend.csv', index=False)
