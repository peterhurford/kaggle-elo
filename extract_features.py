# With help from...
# https://www.kaggle.com/rooshroosh/simple-data-exploration-with-python-lb-3-760
# https://www.kaggle.com/denzo123/a-closer-look-at-date-variables
# https://www.kaggle.com/kailex/tidy-elo-starter-3-715
# https://www.kaggle.com/fabiendaniel/elo-world
# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

import time
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

    print_step('Process Dates')
    for df in [train, test]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
        df['first_active_actual_month'] = df['first_active_month'].dt.month

    print_step('Dummyizing train')
    train = pd.get_dummies(train, columns = ['feature_1', 'feature_2'])
    train = reduce_mem_usage(train)
    print_step('Dummyizing test')
    test = pd.get_dummies(test, columns = ['feature_1', 'feature_2'])
    test = reduce_mem_usage(test)

    print_step('Dummyizing hist 1/3')
    hist_dummies = pd.get_dummies(hist[['category_2', 'category_3']], columns=['category_2', 'category_3'], dummy_na=True)
    print_step('Dummyizing hist 2/3')
    hist = pd.concat([hist, hist_dummies], axis=1)
    print_step('Dummyizing hist 3/3')
    hist = reduce_mem_usage(hist)

    print_step('Dummyizing merch 1/3')
    merch_dummies = pd.get_dummies(merch[['category_2', 'category_3']], columns=['category_2', 'category_3'], dummy_na=True)
    print_step('Dummyizing merch 2/3')
    merch = pd.concat([merch, merch_dummies], axis=1)
    print_step('Dummyizing merch 3/3')
    merch = reduce_mem_usage(merch)

    print_step('Cleaning installments - hist')
    hist['installments_-1'] = (hist['installments'] == -1).astype(int)
    hist['installments_0'] = (hist['installments'] == 0).astype(int)
    hist['installments_1'] = (hist['installments'] == 1).astype(int)
    hist['installments_>1'] = ((hist['installments'] > 1) & (hist['installments'] != 999)).astype(int)
    hist['installments_999'] = (hist['installments'] == 999).astype(int)
    hist['installments'] = hist['installments'].apply(lambda i: 0 if i < 0 or i == 999 else i)

    print_step('Cleaning installments - merch')
    merch['installments_-1'] = (merch['installments'] == -1).astype(int)
    merch['installments_0'] = (merch['installments'] == 0).astype(int)
    merch['installments_1'] = (merch['installments'] == 1).astype(int)
    merch['installments_>1'] = ((merch['installments'] > 1) & (merch['installments'] != 999)).astype(int)
    merch['installments_999'] = (merch['installments'] == 999).astype(int)
    merch['installments'] = merch['installments'].apply(lambda i: 0 if i < 0 or i == 999 else i)

    print_step('Transforming auth')
    agg_fun = {'authorized_flag': ['sum', 'mean']}
    auth_mean = hist.groupby(['card_id']).agg(agg_fun)
    auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
    auth_mean.reset_index(inplace=True)
    auth = hist[hist['authorized_flag'] == 1]
    hist = hist[hist['authorized_flag'] == 0]

    i = 1
    for df in [hist, auth, merch]:
        print_step('Transaction Dates {}/3'.format(i))
        i += 1
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['purchase_month'] = df['purchase_date'].dt.month
        df['purchase_year'] = df['purchase_date'].dt.year
        df['purchase_hour'] = df['purchase_date'].dt.hour
        df['purchase_weekofyear'] = df['purchase_date'].dt.weekofyear
        df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
        df['purchase_weekend'] = (df['purchase_date'].dt.weekday >= 5).astype(int)
        df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']

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
            'category_2_nan': ['sum', 'mean'],
            'category_3_A': ['sum', 'mean'],
            'category_3_B': ['sum', 'mean'],
            'category_3_C': ['sum', 'mean'],
            'category_3_nan': ['sum', 'mean'],
            'category_2': ['nunique'],
            'category_3': ['nunique'],
            'merchant_id': ['nunique'],
            'installments_0': ['sum', 'mean'],
            'installments_-1': ['sum', 'mean'],
            'installments_1': ['sum', 'mean'],
            'installments_>1': ['sum', 'mean'],
            'installments_999': ['sum', 'mean'],
            'merchant_category_id': ['nunique'],
            'state_id': ['nunique'],
            'city_id': ['nunique'],
            'subsector_id': ['nunique'],
            'purchase_amount': ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'skew'],
            'installments': ['sum', 'mean', 'median', 'max', 'min', 'std', 'var'],
            'purchase_year': ['nunique'],
            'purchase_month': ['nunique', 'mean', 'median', 'max', 'min', 'std', 'var', 'skew'],
            'purchase_hour': ['nunique', 'mean', 'std', 'var', 'skew'],
            'purchase_weekofyear': ['nunique', 'mean', 'std', 'var', 'skew'],
            'purchase_dayofweek': ['nunique', 'mean', 'std', 'var', 'skew'],
            'purchase_weekend': ['sum', 'mean'],
            'purchase_date': [np.ptp, 'min', 'max'],
            'month_lag': ['min', 'max'],
            'month_diff': ['mean']
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
            'purchase_amount': ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'skew'],
            'installments': ['count', 'sum', 'mean', 'median', 'min', 'max', 'std'],
        }
        intermediate_group = grouped.agg(agg_func)
        intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
        intermediate_group.reset_index(inplace=True)
        final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
        final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
        final_group.reset_index(inplace=True)
        return final_group

    print_step('Aggregating hist by month')
    hist_month = aggregate_per_month(hist) 
    hist_month.columns = ['hist_month_' + c if c != 'card_id' else c for c in hist_month.columns]

    print_step('Aggregating auth by month')
    auth_month = aggregate_per_month(auth) 
    auth_month.columns = ['auth_month_' + c if c != 'card_id' else c for c in auth_month.columns]

    print_step('Aggregating hist')
    hist = aggregate_transactions(hist)
    hist.columns = ['hist_' + c if c != 'card_id' else c for c in hist.columns]

    print_step('Aggregating auth')
    auth = aggregate_transactions(auth)
    auth.columns = ['auth_' + c if c != 'card_id' else c for c in auth.columns]

    print_step('Aggregating merch')
    merch = aggregate_transactions(merch)
    merch.columns = ['merch_' + c if c != 'card_id' else c for c in merch.columns]

    print_step('Interactions 1/2')
    hist['hist_purchase_date_ptp_over_trans_count'] = hist['hist_purchase_date_ptp'] / hist['hist_transactions_count']
    auth['auth_purchase_date_ptp_over_trans_count'] = auth['auth_purchase_date_ptp'] / auth['auth_transactions_count']
    merch['merch_purchase_date_ptp_over_trans_count'] = merch['merch_purchase_date_ptp'] / merch['merch_transactions_count']
    print_step('Interactions 2/2')
    hist['hist_time_since_now'] = time.mktime(datetime.datetime.now().timetuple()) - hist['hist_purchase_date_max']
    auth['auth_time_since_now'] = time.mktime(datetime.datetime.now().timetuple()) - auth['auth_purchase_date_max']
    merch['merch_time_since_now'] = time.mktime(datetime.datetime.now().timetuple()) - merch['merch_purchase_date_max']


    print_step('Merging 1/12')
    train = pd.merge(train, hist, on='card_id', how='left')
    print_step('Merging 2/12')
    test = pd.merge(test, hist, on='card_id', how='left')

    print_step('Merging 3/12')
    train = pd.merge(train, auth, on='card_id', how='left')
    print_step('Merging 4/12')
    test = pd.merge(test, auth, on='card_id', how='left')

    print_step('Merging 5/12')
    train = pd.merge(train, merch, on='card_id', how='left')
    print_step('Merging 6/12')
    test = pd.merge(test, merch, on='card_id', how='left')

    print_step('Merging 7/12')
    train = pd.merge(train, hist_month, on='card_id', how='left')
    print_step('Merging 8/12')
    test = pd.merge(test, hist_month, on='card_id', how='left')

    print_step('Merging 9/12')
    train = pd.merge(train, auth_month, on='card_id', how='left')
    print_step('Merging 10/12')
    test = pd.merge(test, auth_month, on='card_id', how='left')

    print_step('Merging 11/12')
    train = pd.merge(train, auth_mean, on='card_id', how='left')
    print_step('Merging 12/12')
    test = pd.merge(test, auth_mean, on='card_id', how='left')

    print_step('More Interactions')
    for df in [train, test]:
        df['first_active_month'].fillna(df['first_active_month'].max(), inplace=True)
        df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        df['merch_first_buy'] = (df['merch_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        df['auth_first_buy'] = (df['auth_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        df['total_transactions'] = df['hist_transactions_count'] + df['merch_transactions_count'] + df['auth_transactions_count']
        df['total_purchases'] = df['hist_purchase_amount_sum'] + df['merch_purchase_amount_sum'] + df['auth_purchase_amount_sum']
        df['total_auth_transactions'] = df['merch_transactions_count'] + df['auth_transactions_count']
        df['total_auth_purchases'] = df['merch_purchase_amount_sum'] + df['auth_purchase_amount_sum']

    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~')
    print_step('Saving')
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    save_in_cache('data_with_fe', train, test)
