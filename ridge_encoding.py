# https://www.kaggle.com/tunguz/eloda-with-feature-engineering-and-stacking
# https://www.kaggle.com/tapioca/category-1-2-3-in-transactions was also helpful
import pandas as pd
import numpy as np

from pprint import pprint

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import print_step, rmse, first
from cache import load_cache, save_in_cache, is_in_cache, get_data

    
def runRidge(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Train Ridge')
    model = Ridge(alpha=params.get('alpha', 10))
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model.coef_

def make_svd(tr_vec, te_vec, index, total, label, dimensions=40):
    print_step('SVD {}/{} 1/5 ({})'.format(index, total, label))
    svd = TruncatedSVD(dimensions)
    tr_vec_svd = svd.fit_transform(tr_vec)
    print_step('SVD {}/{} 2/5 ({})'.format(index, total, label))
    te_vec_svd = svd.transform(te_vec)
    print_step('SVD {}/{} 3/5 ({})'.format(index, total, label))
    tr_vec_svd = pd.DataFrame(tr_vec_svd, columns = ['{}_svd_{}'.format(label, i) for i in range(1, dimensions + 1)])
    print_step('SVD {}/{} 4/5 ({})'.format(index, total, label))
    te_vec_svd = pd.DataFrame(te_vec_svd, columns = ['{}_svd_{}'.format(label, i) for i in range(1, dimensions + 1)])
    print_step('SVD {}/{} 5/5 ({})'.format(index, total, label))
    save_in_cache('{}_svd'.format(label), tr_vec_svd, te_vec_svd, verbose=False)
    print('-')
    print(svd.explained_variance_ratio_.sum())
    print('-')
    print(svd.explained_variance_ratio_)
    return None


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test, hist, merch = get_data()
print_step('Subsetting')
target = train['target']
train_id = train['card_id']
test_id = test['card_id']

if not is_in_cache('ridge_encode_feature'):
    print_step('Making feature 1/2')
    train_ = train
    test_ = test
    train_['feature'] = train['feature_1'].astype('str') + train['feature_2'].astype('str')
    print_step('Making feature 2/2')
    test_['feature'] = test['feature_1'].astype('str') + test['feature_2'].astype('str')

    print_step('Dummy Encoding 1/2')
    train_ = pd.get_dummies(train_, columns = ['feature'])
    print_step('Dummy Encoding 2/2')
    test_ = pd.get_dummies(test_, columns = ['feature'])

    print_step('Cleaning 1/2')
    train_ = train_[[c for c in train_.columns if 'feature' in c]].drop(['feature_1', 'feature_2', 'feature_3'], axis=1)
    print_step('Cleaning 1/2')
    test_ = test_[[c for c in test_.columns if 'feature' in c]].drop(['feature_1', 'feature_2', 'feature_3'], axis=1)

    print('~~~~~~~~~~~~~~~~')
    print_step('Run Ridge Encode')
    results = run_cv_model(train_, test_, target, runRidge, {'alpha': 1000}, rmse, 'ridge_encode_feature')
    print(results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False))

    print('~~~~~~~~~~')
    print_step('Cache')
    save_in_cache('ridge_encode_feature', pd.DataFrame({'ridge_encode_feature': results['train'],
                                                        'card_id': train_id}),
                                          pd.DataFrame({'ridge_encode_feature': results['test'],
                                                        'card_id': test_id}))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Prepping submission file')
    submission = pd.DataFrame()
    submission['card_id'] = test_id
    submission['target'] = results['test']
    submission.to_csv('submit/submit_ridge_encode_feature.csv', index=False)
    print_step('Done!')


print('~~~~~~~~~~')
cat_cols = ['city_id', 'category_1', 'category_2', 'installments', 'category_3', 'merchant_category_id', 'subsector_id', 'purchase_amount_bin', 'purchase_hour', 'purchase_dayofweek']
dimensions_lookup = {'city_id': 30, 'category_1': 1, 'category_2': 4, 'installments': 2, 'category_3': 2, 'merchant_category_id': 10, 'subsector_id': 8, 'purchase_amount_bin': 4, 'purchase_hour': 10, 'purchase_dayofweek': 3}
total = len(cat_cols) + 1
print_step('Encoding Vect data 1/{}'.format(total))
tr_vec, te_vec = load_cache('char_vects')
results = run_cv_model(tr_vec, te_vec, target, runRidge, {'alpha': 10000}, rmse, 'ridge_encode_cv')
save_in_cache('ridge_encode_cv', pd.DataFrame({'ridge_encode_cv': results['train']}),
                                 pd.DataFrame({'ridge_encode_cv': results['test']}))
print(tr_vec.shape)
make_svd(tr_vec, te_vec, index=1, total=total, label='char_vects', dimensions=30)

i = 2
total = len(cat_cols)
for col in cat_cols:
    print('~~~~~~~~~~')
    print_step('Encoding Vect data {}/{} ({})'.format(i, total, col))
    tr_vec, te_vec = load_cache('char_vects_{}'.format(col), verbose=False)
    results = run_cv_model(tr_vec, te_vec, target, runRidge, {'alpha': 1000}, rmse, 'ridge_encode_cv_{}'.format(col))
    save_in_cache('ridge_encode_cv_{}'.format(col), pd.DataFrame({'ridge_encode_cv_{}'.format(col): results['train']}),
                                                    pd.DataFrame({'ridge_encode_cv_{}'.format(col): results['test']}))
    print(tr_vec.shape)
    make_svd(tr_vec, te_vec, index=i, total=total, label=col, dimensions=dimensions_lookup[col])
    i += 1

# *Final CV*:         3.8455 (2783 dimensions, 96% explained in 30 components)
# Category3:          3.8462 (49 dimensions, 99% explained in 2 components)
# Installments:       3.8466 (1972 dimensions, 99% explained in 2 components)
# Category1:          3.8478 (6 dimensions, 97% explained in 1 component)
# MerchantCategoryId: 3.8481 (324 dimensions, 91% explained in 10 components)
# Category2:          3.8484 (42 dimensions, 92% explained in 4 components)
# SubsectorId:        3.8492 (41 dimensions, 94% explained in 8 components)
# PurchaseHour:       3.8498 (24 dimensions, 92% explained in 10 components)
# PurchaseDayOfWeek:  3.8503 (7 dimensions, 94% explained in 3 components)
# PurchaseAmountBin:  3.8506 (10 dimensions, 91% explained in 4 components)
# City ID:            3.8517 (308 dimensions, 69% explained in 30 components)
