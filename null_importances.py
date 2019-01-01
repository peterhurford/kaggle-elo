# Adapted from https://www.kaggle.com/fabiendaniel/selecting-features/notebook
# which was adapted from https://www.kaggle.com/ogrellier/feature-selection-with-null-importances

import gc
import time

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

import lightgbm as lgb

from utils import print_step, rmse
from drops import get_drops, save_drops, add_drops
from cache import load_cache, save_in_cache, is_in_cache

if not is_in_cache('data_for_null_importances'):
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data')
    train, test = load_cache('data_with_fe', verbose=False)
    del test; gc.collect()
    features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
    print(train[features].shape)
    drops = get_drops()
    features = [f for f in features if f not in drops]
    train = train[features]
    print(train.shape)
    save_in_cache('data_for_null_importances', train, None)
    del train; gc.collect()


def get_feature_importances(shuffle, label='baseline'):
    print_step('Importances run for {}...'.format(label))
    if is_in_cache('null_importances_{}'.format(label)):
        print_step('Importances run for {} already found. Skipping...'.format(label))
        return None
    train, _ = load_cache('data_for_null_importances', verbose=False)
    print_step('{}: Allocating target...'.format(label))
    if shuffle:
        # Here you could as well use a binomial distribution
        y = train['target'].copy().sample(frac=1.0)
    else:
        y = train['target'].copy()
    
    drops = get_drops(verbose=False)
    features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'target']]
    features = [f for f in features if f not in drops]
    train = train[features]
    print_step('{}: Running LGB...'.format(label))
    lgb_params = {'application': 'regression',
                  'boosting': 'gbdt',
                  'metric': 'rmse',
                  'num_leaves': 16,
                  'max_depth': 8,
                  'learning_rate': 0.01,
                  'bagging_fraction': 0.9,
                  'feature_fraction': 0.8,
                  'lambda_l1': 100,
                  'lambda_l2': 100,
                  'min_data_in_leaf': 40,
                  'verbosity': -1,
                  'data_random_seed': 3,
                  'nthread': 16}
    dtrain = lgb.Dataset(train, label=y)
    clf = lgb.train(params=lgb_params,
                    train_set=dtrain,
                    num_boost_round=2200,
                    valid_sets=[dtrain],
                    verbose_eval=200)

    imp_df = pd.DataFrame()
    imp_df['feature'] = list(features)
    imp_df['importance_gain'] = clf.feature_importance(importance_type='gain')
    imp_df['importance_split'] = clf.feature_importance(importance_type='split')
    imp_df['run'] = label
    save_in_cache('null_importances_{}'.format(label), imp_df, None, verbose=False)
    spent = (time.time() - start) / 60
    print_step('Done with {} of {} (Spent {} min)'.format(label, nb_runs, spent))
    return None

nb_runs = 49
n_nodes = 1
start = time.time()
if n_nodes > 1:
    pool = mp.ProcessingPool(n_nodes)
    pool.map(lambda dat: get_feature_importances(shuffle=dat[0], label=dat[1]),
             list(zip([False if i == 0 else True for i in range(nb_runs + 1)],
                      ['baseline'] + list(range(1, nb_runs + 1)))))
    pool.close()
    pool.join()
    pool.restart()
else:
    list(map(lambda dat: get_feature_importances(shuffle=dat[0], label=dat[1]),
             list(zip([False if i == 0 else True for i in range(nb_runs + 1)],
                      ['baseline'] + list(range(1, nb_runs + 1))))))

print_step('Concatenating...')
actual = load_cache('null_importances_baseline', verbose=False)[0]
dfs = []
for label in list(range(1, nb_runs + 1)):
    dfs.append(load_cache('null_importances_{}'.format(label), verbose=False)[0])
dfs = pd.concat(dfs)

print_step('Calculating scores...')
feature_scores = []
for _f in dfs['feature'].unique():
    f_null_imps_gain = dfs.loc[dfs['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual.loc[actual['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  
    f_null_imps_split = dfs.loc[dfs['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual.loc[actual['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
save_in_cache('null_importances', scores_df, None)
print(scores_df[scores_df['gain_score'] <= 0].sort_values('gain_score'))
bads = list(scores_df[scores_df['gain_score'] <= 0]['feature'].values)
import pdb
pdb.set_trace()
