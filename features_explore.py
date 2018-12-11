import pandas as pd
import numpy as np

from drops import get_drops
from utils import print_step
from cache import load_cache, save_in_cache

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = load_cache('data_with_fe')

print_step('Subsetting')
target = train['target']
train_id = train['card_id']
test_id = test['card_id']

features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'target']]
#drops = get_drops()
#features = [f for f in features if f not in drops]
print(train[features].shape)
print(test[features].shape)

print('~~~~~~~~~~~~')
print_step('Calculating levels')
num_levels = train[features].nunique()
print(num_levels[num_levels < 3])
import pdb
pdb.set_trace()

print('~~~~~~~~~~~~')
print_step('Correlation analysis')
correlation = train[features].copy()
correlation['target'] = target
correlation = correlation.corr()
print('-')
print(correlation['target'].abs().sort_values(ascending=False)[:50])
print('-')
print(correlation['target'].abs().sort_values()[:50])
import pdb
pdb.set_trace()

print('~~~~~~~~~~~~')
print_step('Intercorrelation analysis')
corr = correlation.values
for i in range(correlation.shape[0]):
    for j in range(i+1, correlation.shape[0]):
        if corr[i,j] > 0.96:
            print(correlation.columns[i], ' ', correlation.columns[j], ' ', corr[i,j])
import pdb
pdb.set_trace()
