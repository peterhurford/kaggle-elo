from math import sqrt
from datetime import datetime

import pandas as pd
import numpy as np

from scipy.stats import mode
from collections import Counter

from sklearn.metrics import mean_squared_error, roc_auc_score



def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def univariate_analysis(target, feature):
    score = roc_auc_score(target > 0, feature)
    return 1 - score if score < 0.5 else score


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def first(x):
    return x.iloc[0]

def second(x):
    if len(x) > 1:
        return x.iloc[1]
    else:
        return None

def third(x):
    if len(x) > 2:
        return x.iloc[2]
    else:
        return None

def last(x):
    return x.iloc[-1]

def second_to_last(x):
    if len(x) > 1:
        return x.iloc[-2]
    else:
        return None

def third_to_last(x):
    if len(x) > 2:
        return x.iloc[-3]
    else:
        return None

def most_common(x):
    return mode(x)[0][0]

def num_most_common(x):
    return mode(x)[1][0]

def second_most_common(x):
    commons = Counter(x).most_common(2)
    if len(commons) > 1:
        return commons[1][0]
    else:
        return commons[0][0]

def num_second_most_common(x):
    commons = Counter(x).most_common(2)
    if len(commons) > 1:
        return commons[1][1]
    else:
        return 0
