# Adapted from https://www.kaggle.com/fabiendaniel/selecting-features/notebook
# which was adapted from https://www.kaggle.com/ogrellier/feature-selection-with-null-importances

import pandas as pd
import numpy as np

import lightgbm as lgb

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

#drops = get_drops()
#print('Current drops are {}'.format(', '.join(drops)))
#features = [f for f in features if f not in drops]
print(train[features].shape)
print(test[features].shape)


def get_feature_importances(data, shuffle, seed=None):
    # Shuffle target if required
    y = data['target'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['target'].copy().sample(frac=1.0)
    
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False, silent=True)
    lgb_params = {'application': 'regression',
                  'boosting': 'gbdt',
                  'metric': 'rmse',
                  'num_leaves': 55,
                  'max_depth': 8,
                  'learning_rate': 0.05,
                  'bagging_fraction': 0.94,
                  'feature_fraction': 0.625,
                  'lambda_l1': 79.34,
                  'lambda_l2': 79.39,
                  'min_data_in_leaf': 58,
                  'verbosity': -1,
                  'data_random_seed': 3,
                  'nthread': 4}
    
    # Fit the model
    clf = lgb.train(params=lgb_params,
                    train_set=dtrain,
                    num_boost_round=850)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df['feature'] = list(features)
    imp_df['importance_gain'] = clf.feature_importance(importance_type='gain')
    imp_df['importance_split'] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = rmse(y, clf.predict(data[features]))
    return imp_df

print('~~~~~~~~~~~~~~~~~')
print_step('Get baseline')
# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(data=train, shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 50
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=train, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print_step(dsp)

feature_scores = []
max_features = 300
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
save_in_cache('null_importances', scores_df, None)
print(scores_df[scores_df['gain_score'] <= 0].sort_values('gain_score'))
import pdb
pdb.set_trace()
