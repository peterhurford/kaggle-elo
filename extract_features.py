# With help from...
# https://www.kaggle.com/rooshroosh/simple-data-exploration-with-python-lb-3-760
# https://www.kaggle.com/denzo123/a-closer-look-at-date-variables
# https://www.kaggle.com/kailex/tidy-elo-starter-3-715
# https://www.kaggle.com/fabiendaniel/elo-world
# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

import gc
import time
import datetime

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from utils import print_step, reduce_mem_usage, first, last, second_to_last, most_common, num_most_common, second_most_common, num_second_most_common
from cache import get_data, is_in_cache, save_in_cache


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test, hist, merch = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')

print('~~~~~~~~~~~~~~~')
print_step('Binarizing')
for df in [hist, merch]:
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})

print_step('Sorting 1/2')
hist = hist.sort_values(['card_id', 'purchase_date'])
print_step('Sorting 2/2')
merch = merch.sort_values(['card_id', 'purchase_date'])

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

print_step('Cleaning')
hist['installments'] = hist['installments'].apply(lambda i: 0 if i < 0 or i == 999 else i)
merch['installments'] = merch['installments'].apply(lambda i: 0 if i < 0 or i == 999 else i)
merch['merchant_id'] = merch['merchant_id'].astype('str').fillna('M_ID_missing')
hist['merchant_id'] = hist['merchant_id'].astype('str').fillna('M_ID_missing')
merch['category_3'] = merch['category_3'].astype('str').fillna('missing')
hist['category_3'] = hist['category_3'].astype('str').fillna('missing')

print_step('Transforming auth and fraud')
agg_fun = {'authorized_flag': ['sum', 'mean']}
auth_mean = hist.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)
auth = hist[hist['authorized_flag'] == 1]
fraud = hist[hist['authorized_flag'] == 0]
del hist
gc.collect()

i = 1
for df in [auth, fraud, merch]:
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
        'category_1': ['sum', 'mean', first, last],
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
        'category_2': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'category_3': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'merchant_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'installments_0': ['sum', 'mean'],
        'installments_-1': ['sum', 'mean'],
        'installments_1': ['sum', 'mean'],
        'installments_>1': ['sum', 'mean'],
        'installments_999': ['sum', 'mean'],
        'merchant_category_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'state_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'city_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'subsector_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'purchase_amount': ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'skew', first, last, second_to_last, most_common, num_most_common, second_most_common, num_second_most_common],
        'installments': ['sum', 'mean', 'median', 'max', 'min', 'std', 'var'],
        'purchase_year': ['nunique'],
        'purchase_month': ['nunique', 'mean', 'median', 'max', 'min', 'std', 'var', 'skew', most_common, num_most_common, second_most_common, num_second_most_common],
        'purchase_hour': ['nunique', 'mean', 'std', 'var', 'skew'],
        'purchase_weekofyear': ['nunique', 'mean', 'std', 'var', 'skew'],
        'purchase_dayofweek': ['nunique', 'mean', 'std', 'var', 'skew', most_common, num_most_common, second_most_common, num_second_most_common],
        'purchase_weekend': ['sum', 'mean'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max', most_common, num_most_common, second_most_common, num_second_most_common],
        'month_diff': ['mean']
    }
    agg_items = list(agg_func.items())
    total = len(agg_items) + 2
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 1
    pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
    print_step('Starting a jobs server with %d nodes' % n_nodes)

    def single_transform(index, agg_item):
        print_step('Agg transform {}/{}'.format(index + 1, total))
        transform_df = history.groupby(['card_id']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df

    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    print_step('Agg transform {}/{}'.format(total - 1, total))
    tdf = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    print_step('Agg transform {}/{}'.format(total, total))
    for df in dfs:
        tdf = pd.merge(tdf, df.reset_index(), on='card_id', how='left')

    return tdf

def aggregate_per_month(history):
    agg_func = {
        'merchant_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['count', 'sum', np.ptp, 'mean', 'median', 'min', 'max', 'std', 'skew'],
        'installments': ['count', 'sum', 'mean', 'median', 'min', 'max', 'std'],
    }
    agg_items = list(agg_func.items())
    total = len(agg_items) + 1
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 1
    pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
    print_step('Starting a jobs server with %d nodes' % n_nodes)

    def single_transform(index, agg_item):
        print_step('Agg transform {}/{}'.format(index + 1, total))
        transform_df = history.groupby(['card_id', 'month_lag']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df

    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    print_step('Agg transform {}/{}'.format(total, total))
    tdf = dfs[0].reset_index()
    for df in dfs[1:]:
        tdf = pd.merge(tdf, df.reset_index(), on=['card_id', 'month_lag'], how='left')

    cols = [c for c in tdf.columns if c != 'card_id']
    aggs = ['mean', 'std', 'var', 'skew', 'min', 'max', first, last, second_to_last]
    agg_items = [(c, aggs) for c in cols]
    total = len(agg_items) + 1

    def single_transform(index, agg_item):
        print_step('Agg transform {}/{}'.format(index + 1, total))
        transform_df = tdf.groupby(['card_id']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df

    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    print_step('Agg transform {}/{}'.format(total, total))
    tdf = dfs[0].reset_index()
    for df in dfs[1:]:
        tdf = pd.merge(tdf, df.reset_index(), on='card_id', how='left')
    return tdf

print_step('Aggregating auth by month')
card_chunks = [c for c in auth.groupby('card_id')]
num_chunks = 4
chunk_size = len(card_chunks) // num_chunks
chunk_range = [(chunk_size * i, chunk_size * (i + 1)) for i in range(num_chunks)]
auth_months = []
for i in range(num_chunks):
	start = chunk_range[i][0]
	end = chunk_range[i][1]
	print_step('Aggregating auth by month - Chunk {}/{} - {} to {}'.format(i + 1, num_chunks, start, end))
	df = pd.concat([c[1] for c in card_chunks[start:end]])
	auth_month = aggregate_per_month(df)
	auth_month.columns = ['auth_month_' + c if c != 'card_id' else c for c in auth_month.columns]
	auth_months.append(auth_month)
auth_month = pd.concat(auth_months)
del auth_months
gc.collect()

print_step('Aggregating fraud by month')
fraud_month = aggregate_per_month(fraud) 
fraud_month.columns = ['fraud_month_' + c if c != 'card_id' else c for c in fraud_month.columns]

print_step('Aggregating merch by month')
merch_month = aggregate_per_month(merch) 
merch_month.columns = ['merch_month_' + c if c != 'card_id' else c for c in merch_month.columns]

print_step('Aggregating auth')
auths = []
for i in range(num_chunks):
	start = chunk_range[i][0]
	end = chunk_range[i][1]
	print_step('Aggregating auth - Chunk {}/{} - {} to {}'.format(i + 1, num_chunks, start, end))
	df = pd.concat([c[1] for c in card_chunks[start:end]])
	auth = aggregate_transactions(df)
	auth.columns = ['auth_' + c if c != 'card_id' else c for c in auth.columns]
	auths.append(auth)
auth = pd.concat(auths)
del auths
gc.collect()

print_step('Aggregating fraud')
fraud = aggregate_transactions(fraud)
fraud.columns = ['fraud_' + c if c != 'card_id' else c for c in fraud.columns]

print_step('Aggregating merch')
merch = aggregate_transactions(merch)
merch.columns = ['merch_' + c if c != 'card_id' else c for c in merch.columns]

print_step('Merging 1/12')
train = pd.merge(train, auth, on='card_id', how='left')
print_step('Merging 2/12')
test = pd.merge(test, auth, on='card_id', how='left')
del auth
gc.collect()

print_step('Merging 3/12')
train = pd.merge(train, fraud, on='card_id', how='left')
print_step('Merging 4/12')
test = pd.merge(test, fraud, on='card_id', how='left')
del fraud
gc.collect()

print_step('Merging 5/12')
train = pd.merge(train, merch, on='card_id', how='left')
print_step('Merging 6/12')
test = pd.merge(test, merch, on='card_id', how='left')
del merch
gc.collect()

print_step('Merging 7/12')
train = pd.merge(train, auth_month, on='card_id', how='left')
print_step('Merging 8/12')
test = pd.merge(test, auth_month, on='card_id', how='left')
del auth_month
gc.collect()

print_step('Merging 9/12')
train = pd.merge(train, fraud_month, on='card_id', how='left')
print_step('Merging 10/12')
test = pd.merge(test, fraud_month, on='card_id', how='left')
del fraud_month
gc.collect()

print_step('Merging 9/12')
train = pd.merge(train, merch_month, on='card_id', how='left')
print_step('Merging 10/12')
test = pd.merge(test, merch_month, on='card_id', how='left')
del merch_month
gc.collect()

print_step('Merging 11/12')
train = pd.merge(train, auth_mean, on='card_id', how='left')
print_step('Merging 12/12')
test = pd.merge(test, auth_mean, on='card_id', how='left')
del auth_mean
gc.collect()

i = 1; total = 36
for df in [train, test]:
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['first_active_month'].fillna(df['first_active_month'].max(), inplace=True)
    df['merch_first_buy'] = (df['merch_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['auth_first_buy'] = (df['auth_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_first_buy'] = (df['fraud_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['total_transactions'] = df['auth_transactions_count'] + df['merch_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['total_purchases'] = df['auth_purchase_amount_sum'] + df['merch_purchase_amount_sum']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['total_installments'] = df['auth_installments_sum'] + df['merch_installments_sum']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['total_purchases_over_installments'] = df['total_purchases'] / df['total_installments']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['merch_purchases_over_installments'] = df['merch_purchase_amount_sum'] / df['merch_installments_sum']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['auth_purchases_over_installments'] = df['auth_purchase_amount_sum'] / df['auth_installments_sum']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_purchases_over_installments'] = df['fraud_purchase_amount_sum'] / df['fraud_installments_sum']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['total_purchases_over_transactions'] = df['total_purchases'] / df['total_transactions']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['merch_purchases_over_transactions'] = df['merch_purchase_amount_sum'] / df['merch_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['auth_purchases_over_transactions'] = df['auth_purchase_amount_sum'] / df['auth_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_purchases_over_transactions'] = df['fraud_purchase_amount_sum'] / df['fraud_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['time_to_fraud'] = df['fraud_first_buy'] - df['auth_first_buy']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_purchase_percent'] = df['fraud_purchase_amount_sum'] / (df['fraud_purchase_amount_sum'] + df['auth_purchase_amount_sum'])
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['merch_purchase_percent'] = df['merch_purchase_amount_sum'] / (df['merch_purchase_amount_sum'] + df['auth_purchase_amount_sum'])
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['auth_purchase_date_ptp_over_trans_count'] = df['auth_purchase_date_ptp'] / df['auth_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_purchase_date_ptp_over_trans_count'] = df['fraud_purchase_date_ptp'] / df['fraud_transactions_count']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['auth_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['auth_purchase_date_max']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['fraud_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['fraud_purchase_date_max']
    print_step('Interactions {}/{}'.format(i, total)); i += 1
    df['merch_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['merch_purchase_date_max']
    # Verify month agg worked, first worked, last worked, etc.
    # First minus last
    # Last minus second to last
	# First is same as last (for categorical)
	# Second to last is same as last (for categorical)
    # Most common count / total
    # Second most common count / total
    # Ratio of most common count to second most common count

print_step('Handling categoricals 1/6')
cat_cols = [c[0] for c in list(train.dtypes[train.dtypes == 'object'].items()) if c[0] != 'card_id']
simple_cat_cols = [c for c in cat_cols if 'merchant_id' not in c]
print_step('Handling categoricals 2/6')
train = pd.get_dummies(train, columns = simple_cat_cols)
print_step('Handling categoricals 3/6')
test = pd.get_dummies(test, columns = simple_cat_cols)
print_step('Handling categoricals 4/6')
cat_cols = [c[0] for c in list(train.dtypes[train.dtypes == 'object'].items()) if c[0] != 'card_id']
train['is_train'] = 1
test['is_train'] = 0
tr_te = pd.concat([train, test])
print_step('Handling categoricals 5/6')
for col in cat_cols:
	tr_te[col] = tr_te.groupby(col)[col].transform('count')
# Bayes encode
print_step('Handling categoricals 6/6')
train = tr_te[tr_te['is_train'] == 1]
test = tr_te[tr_te['is_train'] == 0]
train = train.drop(['is_train'], axis=1)
test = test.drop(['is_train', 'target'], axis=1)
del tr_te
gc.collect()

print(train.shape)
print(test.shape)
import pdb
pdb.set_trace()

print('~~~~~~~~~~~')
print_step('Saving')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
save_in_cache('data_with_fe', train, test)
