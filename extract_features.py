# With help from...
# https://www.kaggle.com/rooshroosh/simple-data-exploration-with-python-lb-3-760
# https://www.kaggle.com/denzo123/a-closer-look-at-date-variables
# https://www.kaggle.com/kailex/tidy-elo-starter-3-715
# https://www.kaggle.com/fabiendaniel/elo-world
# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

import datetime

import pandas as pd
import numpy as np

from utils import print_step, reduce_mem_usage
from cache import get_data, is_in_cache, save_in_cache


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test, hist, merch = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')

if not is_in_cache('data_with_fe'):
    print('~~~~~~~~~~~~~~~')
    print_step('Binarizing')
    for df in [hist, merch]:
        for col in ['authorized_flag', 'category_1']:
            df[col] = df[col].map({'Y':1, 'N':0})

    print_step('Train Dates')
    train['first_active_month'] = pd.to_datetime(train['first_active_month'])
    train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
    print_step('Test Dates')
    test['first_active_month'] = pd.to_datetime(test['first_active_month'])
    test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days

    print_step('Dummyizing train')
    train = pd.get_dummies(train, columns = ['feature_1', 'feature_2'])
    train = reduce_mem_usage(train)
    print_step('Dummyizing test')
    test = pd.get_dummies(test, columns = ['feature_1', 'feature_2'])
    test = reduce_mem_usage(test)

    print_step('Dummyizing hist 1/3')
    hist_dummies = pd.get_dummies(hist[['category_2', 'category_3']], columns=['category_2', 'category_3'])
    print_step('Dummyizing hist 2/3')
    hist = pd.concat([hist, hist_dummies], axis=1)
    print_step('Dummyizing hist 3/3')
    hist = reduce_mem_usage(hist)

    print_step('Dummyizing merch 1/3')
    merch_dummies = pd.get_dummies(merch[['category_2', 'category_3']], columns=['category_2', 'category_3'])
    print_step('Dummyizing merch 2/3')
    merch = pd.concat([merch, merch_dummies], axis=1)
    print_step('Dummyizing merch 3/3')
    merch = reduce_mem_usage(merch)

    print_step('Transforming auth')
    agg_fun = {'authorized_flag': ['sum', 'mean']}
    auth_mean = hist.groupby(['card_id']).agg(agg_fun)
    auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
    auth_mean.reset_index(inplace=True)
    auth = hist[hist['authorized_flag'] == 1]
    hist = hist[hist['authorized_flag'] == 0]

    print_step('Transaction Dates 1/6')
    hist['purchase_date'] = pd.to_datetime(hist['purchase_date'])
    print_step('Transaction Dates 2/6')
    auth['purchase_date'] = pd.to_datetime(auth['purchase_date'])
    print_step('Transaction Dates 3/6')
    merch['purchase_date'] = pd.to_datetime(merch['purchase_date'])
    print_step('Transaction Dates 4/6')
    hist['purchase_month'] = hist['purchase_date'].dt.month
    print_step('Transaction Dates 5/6')
    auth['purchase_month'] = auth['purchase_date'].dt.month
    print_step('Transaction Dates 6/6')
    merch['purchase_month'] = merch['purchase_date'].dt.month

    def aggregate_transactions(history):
        history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                          astype(np.int64) * 1e-9
        agg_func = {
            'category_1': ['sum', 'mean'],
            'category_2_1.0': ['sum', 'mean'],
            'category_2_2.0': ['sum', 'mean'],
            'category_2_3.0': ['sum', 'mean'],
            'category_2_4.0': ['sum', 'mean'],
            'category_2_5.0': ['sum', 'mean'],
            'category_3_A': ['sum', 'mean'],
            'category_3_B': ['sum', 'mean'],
            'category_3_C': ['sum', 'mean'],
            'category_2': ['nunique'],
            'category_3': ['nunique'],
            'merchant_id': ['nunique'],
            'merchant_category_id': ['nunique'],
            'state_id': ['nunique'],
            'city_id': ['nunique'],
            'subsector_id': ['nunique'],
            'purchase_amount': ['sum', 'mean', 'median', 'max', 'min', 'std'],
            'installments': ['sum', 'mean', 'median', 'max', 'min', 'std'],
            'purchase_month': ['mean', 'median', 'max', 'min', 'std'],
            'purchase_date': [np.ptp, 'min', 'max'],
            'month_lag': ['min', 'max']
        }
        agg_history = history.groupby(['card_id']).agg(agg_func)
        agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
        agg_history.reset_index(inplace=True)
        df = (history.groupby('card_id')
              .size()
              .reset_index(name='transactions_count'))
        return pd.merge(df, agg_history, on='card_id', how='left')

    def aggregate_per_month(history):
        grouped = history.groupby(['card_id', 'month_lag'])
        agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
        }
        intermediate_group = grouped.agg(agg_func)
        intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
        intermediate_group.reset_index(inplace=True)
        final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
        final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
        final_group.reset_index(inplace=True)
        return final_group

    print_step('Aggregating hist by month')
    final = aggregate_per_month(hist) 

    print_step('Aggregating hist')
    hist = aggregate_transactions(hist)
    hist.columns = ['hist_' + c if c != 'card_id' else c for c in hist.columns]

    print_step('Aggregating auth')
    auth = aggregate_transactions(auth)
    auth.columns = ['auth_' + c if c != 'card_id' else c for c in auth.columns]

    print_step('Aggregating merch')
    merch = aggregate_transactions(merch)
    merch.columns = ['merch_' + c if c != 'card_id' else c for c in merch.columns]

    print_step('Merging 1/10')
    train = pd.merge(train, hist, on='card_id', how='left')
    print_step('Merging 2/10')
    test = pd.merge(test, hist, on='card_id', how='left')

    print_step('Merging 3/10')
    train = pd.merge(train, auth, on='card_id', how='left')
    print_step('Merging 4/10')
    test = pd.merge(test, auth, on='card_id', how='left')

    print_step('Merging 5/10')
    train = pd.merge(train, merch, on='card_id', how='left')
    print_step('Merging 6/10')
    test = pd.merge(test, merch, on='card_id', how='left')

    print_step('Merging 7/10')
    train = pd.merge(train, final, on='card_id', how='left')
    print_step('Merging 8/10')
    test = pd.merge(test, final, on='card_id', how='left')

    print_step('Merging 9/10')
    train = pd.merge(train, auth_mean, on='card_id', how='left')
    print_step('Merging 10/10')
    test = pd.merge(test, auth_mean, on='card_id', how='left')

    print('~~~~~~~~~~~')
    print_step('Saving')
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    save_in_cache('data_with_fe', train, test)
