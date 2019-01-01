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

from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix, hstack

from utils import print_step, reduce_mem_usage, first, second, third, last, second_to_last, third_to_last, most_common, num_most_common, second_most_common, num_second_most_common
from cache import get_data, is_in_cache, save_in_cache, load_cache


if not is_in_cache('clean') and not is_in_cache('auth_clean') and not is_in_cache('fraud_clean') and not is_in_cache('merch_clean'):
    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data')
    train, test, hist, merch = get_data()

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
    print_step('Dummyizing test')
    test = pd.get_dummies(test, columns = ['feature_1', 'feature_2'])

    print_step('Dummyizing hist 1/3')
    hist['purchase_amount_bin'] = pd.qcut(hist['purchase_amount'], 10, labels=range(10))
    print_step('Dummyizing hist 2/3')
    hist_dummies = pd.get_dummies(hist[['category_2', 'category_3', 'purchase_amount_bin']], columns=['category_2', 'category_3', 'purchase_amount_bin'], dummy_na=True)
    print_step('Dummyizing hist 3/3')
    hist = pd.concat([hist, hist_dummies], axis=1)

    print_step('Dummyizing merch 1/3')
    merch['purchase_amount_bin'] = pd.qcut(hist['purchase_amount'], 10, labels=range(10))
    print_step('Dummyizing merch 2/3')
    merch_dummies = pd.get_dummies(merch[['category_2', 'category_3', 'purchase_amount_bin']], columns=['category_2', 'category_3', 'purchase_amount_bin'], dummy_na=True)
    print_step('Dummyizing merch 3/3')
    merch = pd.concat([merch, merch_dummies], axis=1)

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
    save_in_cache('auth_mean', auth_mean, None)
    auth = hist[hist['authorized_flag'] == 1]
    fraud = hist[hist['authorized_flag'] == 0]
    del hist
    gc.collect()

    i = 1
    for df in [auth, fraud, merch]:
        print_step('Transaction Dates {}/3 1/8'.format(i))
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        print_step('Transaction Dates {}/3 2/8'.format(i))
        df['purchase_month'] = df['purchase_date'].dt.month
        print_step('Transaction Dates {}/3 3/8'.format(i))
        df['purchase_year'] = df['purchase_date'].dt.year
        print_step('Transaction Dates {}/3 4/8'.format(i))
        df['purchase_hour'] = df['purchase_date'].dt.hour
        print_step('Transaction Dates {}/3 5/8'.format(i))
        df['purchase_weekofyear'] = df['purchase_date'].dt.weekofyear
        print_step('Transaction Dates {}/3 6/8'.format(i))
        df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
        print_step('Transaction Dates {}/3 7/8'.format(i))
        df['purchase_weekend'] = (df['purchase_date'].dt.weekday >= 5).astype(int)
        print_step('Transaction Dates {}/3 8/8'.format(i))
        df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']
        i += 1
    save_in_cache('clean', train, test)
    auth = auth.reset_index().drop('index', axis=1)
    save_in_cache('auth_clean', auth, None)
    fraud = fraud.reset_index().drop('index', axis=1)
    save_in_cache('fraud_clean', fraud, None)
    merch = merch.reset_index().drop('index', axis=1)
    save_in_cache('merch_clean', merch, None)


if not is_in_cache('char_vects'):
    cat_cols = ['city_id', 'category_1', 'category_2', 'installments', 'category_3', 'merchant_category_id', 'subsector_id', 'purchase_amount_bin', 'purchase_hour', 'purchase_dayofweek']
    simple_cat_cols = ['purchase_dayofweek', 'purchase_amount_bin', 'category_1']
    low_card_cat_cols = ['category_3', 'installments']
    high_card_cat_cols = ['city_id', 'merchant_category_id', 'purchase_hour', 'subsector_id', 'purchase_dayofweek', 'purchase_amount_bin']
    total = len(cat_cols)

    def single_transform(index, col):
        if is_in_cache('auth_stream_{}'.format(index + 1)):
            print_step('Streaming {}/{} ({}) -- already done, skipped!'.format(index + 1, total, col))
            return None
        print_step('Streaming {}/{} ({})'.format(index + 1, total, col))
        auth, _ = load_cache('auth_clean', verbose=False)
        auth = auth[['card_id'] + cat_cols]
        auth = auth.groupby(['card_id'])
        print_step('Agging {}/{} ({})'.format(index + 1, total, col))
        auth = auth.agg({col: lambda xs: ' '.join([str(x) for x in xs])}).reset_index()
        print_step('Done with {}/{} ({})'.format(index + 1, total, col))
        save_in_cache('auth_stream_{}'.format(index + 1), auth, None)
        return None

    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool = mp.ProcessingPool(n_nodes)
    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(cat_cols))
    pool.close()
    pool.join()
    pool.restart()

    print_step('Merging 1/5 1/{}'.format(total))
    auth_cga = load_cache('auth_stream_{}'.format(1), verbose=False)[0].reset_index()
    for i in range(2, total + 1):
        print_step('Merging 1/5 {}/{}'.format(i, total))
        auth_cga = pd.merge(auth_cga, load_cache('auth_stream_{}'.format(i), verbose=False)[0].reset_index(), on=['card_id'], how='left')

    print_step('Merging 2/5')
    train, test = load_cache('clean', verbose=False)
    print_step('Merging 3/5')
    auth_cga_train = train[['card_id', 'target']].merge(auth_cga, on='card_id')
    print_step('Merging 4/5')
    auth_cga_test = test[['card_id']].merge(auth_cga, on='card_id')
    print_step('Cleaning 1/2')
    auth_cga_train.drop(['index_x', 'index_y'], axis=1, inplace=True)
    print_step('Cleaning 2/2')
    auth_cga_test.drop(['index_x', 'index_y'], axis=1, inplace=True)
    save_in_cache('auth_cga', auth_cga_train, auth_cga_test)

    def single_transform2(index, col):
        auth_cga_train, auth_cga_test = load_cache('auth_cga', verbose=False)
        print_step('Count vectorizing {}/{} ({})'.format(index + 1, total, col))
        if col in high_card_cat_cols:
            ngram_max = 1
        elif col in low_card_cat_cols:
            ngram_max = 3
        else:
            ngram_max = 2
        if col in simple_cat_cols:
            cv = CountVectorizer(stop_words=None, min_df=0, ngram_range=(1, ngram_max), analyzer='char')
            train_vect = cv.fit_transform(auth_cga_train[col].str.replace(' ', ''))
            test_vect = cv.transform(auth_cga_test[col].str.replace(' ', ''))
        else:
            cv = CountVectorizer(stop_words=None, min_df=0, ngram_range=(1, ngram_max), tokenizer=lambda x: x.split(' '))
            train_vect = cv.fit_transform(auth_cga_train[col])
            test_vect = cv.transform(auth_cga_test[col])
        save_in_cache('char_vects_{}'.format(col), train_vect, test_vect)
        return None
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool.close()
    pool.join()
    pool.restart()
    pool = mp.ProcessingPool(n_nodes)
    dfs = pool.map(lambda dat: single_transform2(dat[0], dat[1]), enumerate(cat_cols))

    total = len(cat_cols)
    print_step('Merging 5/5 1/{} ({})'.format(total, cat_cols[0]))
    train_vects, test_vects = load_cache('char_vects_{}'.format(cat_cols[0]), verbose=False)
    print(train_vects.shape)
    print(test_vects.shape)
    for i in range(1, total):
        print_step('Merging 5/5 {}/{} ({})'.format(i + 1, total, cat_cols[i]))
        train_vect, test_vect = load_cache('char_vects_{}'.format(cat_cols[i]), verbose=False)
        train_vects = hstack((train_vects, train_vect))
        print(train_vects.shape)
        test_vects = hstack((test_vects, test_vect))
        print(test_vects.shape)
    print_step('Saving char vects in cache...')
    train_vects = csr_matrix(train_vects)
    test_vects = csr_matrix(test_vects)
    save_in_cache('char_vects', train_vects, test_vects)


def aggregate_transactions(location):
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
        'merchant_category_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'state_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'city_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'subsector_id': ['nunique', first, last, most_common, num_most_common, second_most_common, num_second_most_common],
        'purchase_amount': ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'skew', first, second, last, second_to_last, most_common, num_most_common, second_most_common, num_second_most_common],
        'purchase_amount_bin_0.0': ['sum', 'mean'],
        'purchase_amount_bin_1.0': ['sum', 'mean'],
        'purchase_amount_bin_2.0': ['sum', 'mean'],
        'purchase_amount_bin_3.0': ['sum', 'mean'],
        'purchase_amount_bin_4.0': ['sum', 'mean'],
        'purchase_amount_bin_5.0': ['sum', 'mean'],
        'purchase_amount_bin_6.0': ['sum', 'mean'],
        'purchase_amount_bin_7.0': ['sum', 'mean'],
        'purchase_amount_bin_8.0': ['sum', 'mean'],
        'purchase_amount_bin_9.0': ['sum', 'mean'],
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

    def single_transform(index, agg_item):
        print_step('Agg transform {}/{}'.format(index + 1, total))
        history, _ = load_cache(location, verbose=False)
        history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                          astype(np.int64) * 1e-9
        transform_df = history.groupby(['card_id']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df
    agg_items = list(agg_func.items())
    total = len(agg_items) + 2
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool = mp.ProcessingPool(n_nodes)
    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.restart()
    print_step('Agg transform {}/{}'.format(total - 1, total))
    history, _ = load_cache(location, verbose=False)
    tdf = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    print_step('Agg transform {}/{}'.format(total, total))
    for df in dfs:
        tdf = pd.merge(tdf, df.reset_index(), on='card_id', how='left')
    return tdf

def aggregate_per_month(location):
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
        'category_2': ['nunique', num_most_common, num_second_most_common],
        'category_3': ['nunique', num_most_common, num_second_most_common],
        'merchant_id': ['nunique', num_most_common, num_second_most_common],
        'merchant_category_id': ['nunique', num_most_common, num_second_most_common],
        'state_id': ['nunique', num_most_common, num_second_most_common],
        'city_id': ['nunique', num_most_common, num_second_most_common],
        'purchase_amount': ['count', 'sum', np.ptp, 'mean', 'median', 'min', 'max', 'std', 'skew'],
        'purchase_amount_bin_0.0': ['sum', 'mean'],
        'purchase_amount_bin_1.0': ['sum', 'mean'],
        'purchase_amount_bin_2.0': ['sum', 'mean'],
        'purchase_amount_bin_3.0': ['sum', 'mean'],
        'purchase_amount_bin_4.0': ['sum', 'mean'],
        'purchase_amount_bin_5.0': ['sum', 'mean'],
        'purchase_amount_bin_6.0': ['sum', 'mean'],
        'purchase_amount_bin_7.0': ['sum', 'mean'],
        'purchase_amount_bin_8.0': ['sum', 'mean'],
        'purchase_amount_bin_9.0': ['sum', 'mean'],
        'purchase_dayofweek': ['nunique', 'mean', 'std', 'var', 'skew', num_most_common, num_second_most_common],
        'purchase_weekend': ['sum', 'mean'],
        'installments': ['count', 'sum', 'mean', 'median', 'min', 'max', 'std'],
    }

    def single_transform(index, agg_item):
        history, _ = load_cache(location, verbose=False)
        print_step('Agg transform {}/{}'.format(index + 1, total))
        transform_df = history.groupby(['card_id', 'month_lag']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df
    agg_items = list(agg_func.items())
    total = len(agg_items) + 1
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool = mp.ProcessingPool(n_nodes)
    dfs = pool.map(lambda dat: single_transform(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.restart()

    print_step('Agg transform {}/{}'.format(total, total))
    tdf = dfs[0].reset_index()
    for df in dfs[1:]:
        tdf = pd.merge(tdf, df.reset_index(), on=['card_id', 'month_lag'], how='left')
    save_in_cache('month_tdf', tdf, None)

    cols = [c for c in tdf.columns if c != 'card_id']
    aggs = ['mean', 'std', 'var', 'skew', 'min', 'max', first, second, third, last, second_to_last, third_to_last]
    agg_items = [(c, aggs) for c in cols]
    total = len(agg_items) + 1

    def single_transform2(index, agg_item):
        tdf, _ = load_cache('month_tdf', verbose=False)
        print_step('Agg transform {}/{}'.format(index + 1, total))
        transform_df = tdf.groupby(['card_id']).agg({agg_item[0]: agg_item[1]})
        transform_df.columns = ['_'.join(col).strip() for col in transform_df.columns.values if col != 'card_id']
        return transform_df
    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool = mp.ProcessingPool(n_nodes)
    dfs = pool.map(lambda dat: single_transform2(dat[0], dat[1]), enumerate(agg_items))
    pool.close()
    pool.join()
    pool.restart()

    print_step('Agg transform {}/{}'.format(total, total))
    tdf = dfs[0].reset_index()
    for df in dfs[1:]:
        tdf = pd.merge(tdf, df.reset_index(), on='card_id', how='left')
    tdf = tdf[[c for c in tdf.columns if 'month_lag' not in c]]
    return tdf


if not is_in_cache('auth_month'):
    print_step('Aggregating auth by month')
    auth, _ = load_cache('auth_clean', verbose=False)
    card_chunks = [c for c in auth.groupby('card_id')]
    num_chunks = 8
    chunk_size = len(card_chunks) // num_chunks
    chunk_range = [(chunk_size * i, chunk_size * (i + 1)) for i in range(num_chunks)]
    auth_months = []
    for i in range(num_chunks):
        start = chunk_range[i][0]
        end = chunk_range[i][1]
        print_step('Aggregating auth by month - Chunk {}/{} - {} to {}'.format(i + 1, num_chunks, start, end))
        df = pd.concat([c[1] for c in card_chunks[start:end]]).reset_index().drop('index', axis=1)
        save_in_cache('auth_month_chunk_{}'.format(i), df, None)
        auth_month = aggregate_per_month('auth_month_chunk_{}'.format(i))
        auth_month.columns = ['auth_month_' + c if c != 'card_id' else c for c in auth_month.columns]
        auth_months.append(auth_month)
    auth_month = pd.concat(auth_months)
    del auth_months
    gc.collect()
    auth_month = auth_month.reset_index().drop('index', axis=1)
    save_in_cache('auth_month', auth_month, None)

if not is_in_cache('fraud_month'):
    print_step('Aggregating fraud by month')
    fraud_month = aggregate_per_month('fraud_clean')
    fraud_month.columns = ['fraud_month_' + c if c != 'card_id' else c for c in fraud_month.columns]
    fraud_month = fraud_month.reset_index().drop('index', axis=1)
    save_in_cache('fraud_month', fraud_month, None)


if not is_in_cache('auth_agg'):
    print_step('Aggregating auth')
    auth, _ = load_cache('auth_clean', verbose=False)
    card_chunks = [c for c in auth.groupby('card_id')]
    num_chunks = 8
    chunk_size = len(card_chunks) // num_chunks
    chunk_range = [(chunk_size * i, chunk_size * (i + 1)) for i in range(num_chunks)]
    auths = []
    for i in range(num_chunks):
        start = chunk_range[i][0]
        end = chunk_range[i][1]
        print_step('Aggregating auth - Chunk {}/{} - {} to {}'.format(i + 1, num_chunks, start, end))
        df = pd.concat([c[1] for c in card_chunks[start:end]]).reset_index().drop('index', axis=1)
        save_in_cache('auth_chunk_{}'.format(i), df, None)
        auth = aggregate_transactions('auth_chunk_{}'.format(i))
        auth.columns = ['auth_' + c if c != 'card_id' else c for c in auth.columns]
        auths.append(auth)
    auth = pd.concat(auths)
    auth = auth.reset_index().drop('index', axis=1)
    save_in_cache('auth_agg', auth, None)
    del auths
    gc.collect()
else:
    auth, _ = load_cache('auth_agg')


if not is_in_cache('fraud_agg'):
    print_step('Aggregating fraud')
    fraud = aggregate_transactions('fraud_clean')
    fraud.columns = ['fraud_' + c if c != 'card_id' else c for c in fraud.columns]
    fraud = fraud.reset_index().drop('index', axis=1)
    save_in_cache('fraud_agg', fraud, None)
else:
    fraud, _ = load_cache('fraud_agg')


if not is_in_cache('merch_agg'):
    print_step('Aggregating merch')
    merch = aggregate_transactions('merch_clean')
    merch.columns = ['merch_' + c if c != 'card_id' else c for c in merch.columns]
    merch = merch.reset_index().drop('index', axis=1)
    save_in_cache('merch_agg', merch, None)
else:
    merch, _ = load_cache('merch_agg')


if not is_in_cache('data_with_fe'):
    print_step('Merging 1/12')
    train, test = load_cache('clean')
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
    auth_month, _ = load_cache('auth_month')
    train = pd.merge(train, auth_month, on='card_id', how='left')
    print_step('Merging 8/12')
    test = pd.merge(test, auth_month, on='card_id', how='left')
    del auth_month
    gc.collect()

    print_step('Merging 9/12')
    auth_month, _ = load_cache('fraud_month')
    train = pd.merge(train, fraud_month, on='card_id', how='left')
    print_step('Merging 10/12')
    test = pd.merge(test, fraud_month, on='card_id', how='left')
    del auth_month
    gc.collect()

    print_step('Merging 11/12')
    auth_mean, _ = load_cache('auth_mean')
    train = pd.merge(train, auth_mean, on='card_id', how='left')
    print_step('Merging 12/12')
    test = pd.merge(test, auth_mean, on='card_id', how='left')
    del auth_mean
    gc.collect()

    i = 1; total = 68
    na_max = train['first_active_month'].max()
    train['first_active_month'].fillna(na_max, inplace=True)
    test['first_active_month'].fillna(na_max, inplace=True)
    for df in [train, test]:
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_first_buy'] = (df['merch_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_first_buy'] = (df['auth_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_first_buy'] = (df['fraud_purchase_date_min'] - df['first_active_month'].apply(lambda t: time.mktime(t.timetuple())))
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'total_transactions'] = df['auth_transactions_count'] + df['merch_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'total_purchases'] = df['auth_purchase_amount_sum'] + df['merch_purchase_amount_sum']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'total_installments'] = df['auth_installments_sum'] + df['merch_installments_sum']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'total_purchases_over_installments'] = df['total_purchases'] / df['total_installments']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_purchases_over_installments'] = df['merch_purchase_amount_sum'] / df['merch_installments_sum']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_purchases_over_installments'] = df['auth_purchase_amount_sum'] / df['auth_installments_sum']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_purchases_over_installments'] = df['fraud_purchase_amount_sum'] / df['fraud_installments_sum']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'total_purchases_over_transactions'] = df['total_purchases'] / df['total_transactions']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_purchases_over_transactions'] = df['merch_purchase_amount_sum'] / df['merch_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_purchases_over_transactions'] = df['auth_purchase_amount_sum'] / df['auth_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_purchases_over_transactions'] = df['fraud_purchase_amount_sum'] / df['fraud_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'time_to_fraud'] = df['fraud_first_buy'] - df['auth_first_buy']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_purchase_percent'] = df['fraud_purchase_amount_sum'] / (df['fraud_purchase_amount_sum'] + df['auth_purchase_amount_sum'])
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_purchase_percent'] = df['merch_purchase_amount_sum'] / (df['merch_purchase_amount_sum'] + df['auth_purchase_amount_sum'])
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_purchase_date_ptp_over_trans_count'] = df['auth_purchase_date_ptp'] / df['auth_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_purchase_date_ptp_over_trans_count'] = df['fraud_purchase_date_ptp'] / df['fraud_transactions_count']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['auth_purchase_date_max']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['fraud_purchase_date_max']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_time_since_last'] = time.mktime(datetime.datetime.now().timetuple()) - df['merch_purchase_date_max']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        # TODO: Below can eventually be replaced with group by stats on city_id and merchant_id (and maybe other ids)
        df.loc[:, 'auth_merchants_per_city'] = df['auth_merchant_id_nunique'] / df['auth_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_merchants_per_city'] = df['fraud_merchant_id_nunique'] / df['fraud_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_merchants_per_city'] = df['merch_merchant_id_nunique'] / df['merch_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_transactions_per_city'] = df['auth_transactions_count'] / df['auth_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_transactions_per_city'] = df['fraud_transactions_count'] / df['fraud_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_transactions_per_city'] = df['merch_transactions_count'] / df['merch_city_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_transactions_per_merchant'] = df['auth_transactions_count'] / df['auth_merchant_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_transactions_per_merchant'] = df['fraud_transactions_count'] / df['fraud_merchant_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_transactions_per_merchant'] = df['merch_transactions_count'] / df['merch_merchant_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'auth_purchase_per_merchant'] = df['auth_purchase_amount_sum'] / df['auth_merchant_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'fraud_purchase_per_merchant'] = df['fraud_purchase_amount_sum'] / df['fraud_merchant_id_nunique']
        print_step('Interactions {}/{}'.format(i, total)); i += 1
        df.loc[:, 'merch_purchase_per_merchant'] = df['merch_purchase_amount_sum'] / df['merch_merchant_id_nunique']

    cols = list(train.columns.values)
    firsts = [c for c in cols if c[-5:] == 'first']
    numeric_firsts = train[firsts].dtypes[train[firsts].dtypes == 'float64'].index.values
    numeric_firsts = [c for c in numeric_firsts if 'category' not in c and 'id' not in c]
    i = 1; total = len(numeric_firsts)
    for c in numeric_firsts:
        print_step("Aggregating firsts and lasts {}/{} ({})".format(i, total, c.replace('_first', ''))); i += 1
        train.loc[:, c.replace('_first', '_last_m_first')] = train[c.replace('_first', '_last')] - train[c]
        test.loc[:, c.replace('_first', '_last_m_first')] = test[c.replace('_first', '_last')] - test[c]
        train.loc[:, c.replace('_first', '_last_m_2last')] = train[c.replace('_first', '_last')] - train[c.replace('_first', '_second_to_last')]
        test.loc[:, c.replace('_first', '_last_m_2last')] = test[c.replace('_first', '_last')] - test[c.replace('_first', '_second_to_last')]
        train.loc[:, c.replace('_first', '_second_m_first')] = train[c.replace('_first', '_second')] - train[c]
        test.loc[:, c.replace('_first', '_second_m_first')] = test[c.replace('_first', '_second')] - test[c]
        if c.replace('_first', '_third_to_last') in train.columns:
            train.loc[:, c.replace('_first', '_last_m_3last')] = train[c.replace('_first', '_last')] - train[c.replace('_first', '_third_to_last')]
            test.loc[:, c.replace('_first', '_last_m_3last')] = test[c.replace('_first', '_last')] - test[c.replace('_first', '_third_to_last')]
            train.loc[:, c.replace('_first', '_2last_m_3last')] = train[c.replace('_first', '_last')] - train[c.replace('_first', '_second_to_last')]
            test.loc[:, c.replace('_first', '_2last_m_3last')] = test[c.replace('_first', '_second_to_last')] - test[c.replace('_first', '_third_to_last')]
            train.loc[:, c.replace('_first', '_third_m_first')] = train[c.replace('_first', '_third')] - train[c]
            test.loc[:, c.replace('_first', '_third_m_first')] = test[c.replace('_first', '_third')] - test[c]
            train.loc[:, c.replace('_first', '_third_m_second')] = train[c.replace('_first', '_third')] - train[c.replace('_first', '_second')]
            test.loc[:, c.replace('_first', '_third_m_second')] = test[c.replace('_first', '_third')] - test[c.replace('_first', '_second')]
            train.loc[:, c.replace('_first', '_first3sum')] = train[c] + train[c.replace('_first', '_second')] + train[c.replace('_first', '_third')]
            train.loc[:, c.replace('_first', '_last3sum')] = train[c.replace('_first', '_last')] + train[c.replace('_first', '_second_to_last')] + train[c.replace('_first', '_third_to_last')]
            train.loc[:, c.replace('_first', '_first3_last3_ratio')] = train[c.replace('_first', '_first3sum')] / train[c.replace('_first', '_last3sum')]

    i = 1; total = 37
    for section in ['auth', 'fraud', 'merch']:
        most_common_cols = [c for c in cols if 'num_most_common' in c and 'common_' not in c and section in c]
        for col in most_common_cols:
            print_step('Aggregating most commons {}/{} ({})'.format(i, total, col)); i += 1
            train.loc[:, col.replace('_num_most_common', '_percent_most_common')] = train[col] / train['{}_transactions_count'.format(section)]
            test.loc[:, col.replace('_num_most_common', '_percent_most_common')] = test[col] / test['{}_transactions_count'.format(section)]
            train.loc[:, col.replace('_num_most_common', '_percent_second_most_common')] = train[col.replace('_num_most', '_num_second_most')] / train['{}_transactions_count'.format(section)]
            test.loc[:, col.replace('_num_most_common', '_percent_second_most_common')] = test[col.replace('_num_most', '_num_second_most')] / test['{}_transactions_count'.format(section)]
            train.loc[:, col.replace('_num_most_common', '_most_2most_ratio')] = train[col.replace('_num_most_common', '_percent_most_common')] / train[col.replace('_num_most_common', '_percent_second_most_common')]
            test.loc[:, col.replace('_num_most_common', '_most_2most_ratio')] = test[col.replace('_num_most_common', '_percent_most_common')] / test[col.replace('_num_most_common', '_percent_second_most_common')]


    print_step('Handling categoricals 1/7')
    cat_cols = [c[0] for c in list(train.dtypes[train.dtypes == 'object'].items()) if c[0] != 'card_id']
    cat_cols = [c for c in cat_cols if 'most_common' in c and 'second' not in c]
    simple_cat_cols = [c for c in cat_cols if 'merchant_id' not in c]
    print_step('Handling categoricals 2/7')
    train = pd.get_dummies(train, columns = simple_cat_cols)
    print_step('Handling categoricals 3/7')
    test = pd.get_dummies(test, columns = simple_cat_cols)
    print_step('Handling categoricals 4/7')
    train.loc[:, 'is_train'] = 1
    test.loc[:, 'is_train'] = 0
    tr_te = pd.concat([train, test])
    print_step('Handling categoricals 5/7')
    complex_cat_cols = list(set(cat_cols) - set(simple_cat_cols))
    for col in complex_cat_cols:
        tr_te.loc[:, col] = tr_te.groupby(col)[col].transform('count')
    print_step('Handling categoricals 6/7')
    train = tr_te[tr_te['is_train'] == 1]
    test = tr_te[tr_te['is_train'] == 0]
    train = train.drop(['is_train'], axis=1)
    test = test.drop(['is_train', 'target'], axis=1)
    del tr_te
    gc.collect()
    print_step('Handling categoricals 7/7')
    cat_cols = [c[0] for c in list(train.dtypes[train.dtypes == 'object'].items()) if c[0] != 'card_id']
    train.drop(cat_cols, axis=1, inplace=True)
    test.drop(cat_cols, axis=1, inplace=True)
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~')
    print_step('Saving')
    save_in_cache('data_with_fe', train, test)
