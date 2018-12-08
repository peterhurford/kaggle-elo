# https://www.kaggle.com/tunguz/eloda-with-feature-engineering-and-stacking
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import print_step, rmse
from cache import load_cache, save_in_cache


def runRidge(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Train Ridge')
    model = Ridge(alpha=100)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model.coef_


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

# TODO: Fix
print_step('Fill NA 1/2')
train_ = train.copy()
train_ = train_[features]
train_.fillna((train_.mean()), inplace=True)
print_step('Fill NA 2/2')
test_ = test.copy()
test_ = test_[features]
test_.fillna((test_.mean()), inplace=True)

print('~~~~~~~~~~~~')
print_step('Run Ridge')
results = run_cv_model(train_, test_, target, runRidge, {}, rmse, 'ridge')
results['importance']['abs_value'] = abs(results['importance']['importance'])
print(results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False))
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('ridge', pd.DataFrame({'ridge': results['train']}),
                       pd.DataFrame({'ridge': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = results['test']
submission.to_csv('submit/submit_ridge.csv', index=False)
print_step('Done!')
# [2018-12-07 12:30:24.039155] ridge cv scores : [3.8680803389966316, 3.6487654664807163, 3.8000114790872823, 3.7592781915190057, 3.793188182580381]
# [2018-12-07 12:30:24.041314] ridge mean cv score : 3.773864731732803
# [2018-12-07 12:30:24.041617] ridge std cv score : 0.07182788055343188
# [2018-12-07 12:30:24.062169] ridge final cv score : 3.774548075434262
