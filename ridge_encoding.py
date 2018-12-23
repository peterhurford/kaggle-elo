# https://www.kaggle.com/tunguz/eloda-with-feature-engineering-and-stacking
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import print_step, rmse
from cache import load_cache, save_in_cache, get_data

    
def runRidge(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Train Ridge')
    model = Ridge(alpha=10)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model.coef_


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test, hist, merch = get_data()

print_step('Making feature 1/2')
train['feature'] = train['feature_1'].astype('str') + train['feature_2'].astype('str')
print_step('Making feature 2/2')
test['feature'] = test['feature_1'].astype('str') + test['feature_2'].astype('str')

print_step('Dummy Encoding 1/2')
train = pd.get_dummies(train, columns = ['feature'])
print_step('Dummy Encoding 2/2')
test = pd.get_dummies(test, columns = ['feature'])

print_step('Subsetting')
target = train['target']
train_id = train['card_id']
test_id = test['card_id']

print_step('Cleaning 1/2')
train = train[[c for c in train.columns if 'feature' in c]].drop(['feature_1', 'feature_2', 'feature_3'], axis=1)
print_step('Cleaning 1/2')
test = test[[c for c in test.columns if 'feature' in c]].drop(['feature_1', 'feature_2', 'feature_3'], axis=1)

print('~~~~~~~~~~~~~~~~')
print_step('Run Ridge Encode')
results = run_cv_model(train, test, target, runRidge, {}, rmse, 'ridge')
print(results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False))
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('ridge_encode_feature', pd.DataFrame({'ridge_encode_feature': results['train']}),
                                      pd.DataFrame({'ridge_encode_feature': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = results['test']
submission.to_csv('submit/submit_ridge_encode_feature.csv', index=False)
print_step('Done!')

print('~~~~~~~~~~')
print_step('Munging Vect data 1/6')
hist['authorized_flag'] = hist['authorized_flag'].map({'Y':1, 'N':0})
print_step('Munging Vect data 2/6')
auth = hist[hist['authorized_flag'] == 1]
print_step('Munging Vect data 3/6')
auth_train = auth.merge(train[['card_id', 'target']], on='card_id')
print_step('Munging Vect data 4/6')
auth_test = auth.merge(test[['card_id']], on='card_id')
print_step('Munging Vect data 5/6')
auth_target = auth_train.groupby('card_id')[['card_id', 'target']].mean().reset_index()
print_step('Munging Vect data 6/6')
tr_vec, te_vec = load_cache('char_vects')
print_step('Encoding Vect data')
results = run_cv_model(tr_vec, te_vec, auth_target['target'], runRidge, {}, rmse, 'ridge')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('ridge_encode_cv', pd.DataFrame({'ridge_encode_cv': results['train']}),
                                 pd.DataFrame({'ridge_encode_cv': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['card_id'] = test_id
submission['target'] = results['test']
submission.to_csv('submit/submit_ridge_encode_cv.csv', index=False)
print_step('Done!')
