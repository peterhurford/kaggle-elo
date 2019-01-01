# Stratified fold strategy from https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
import gc

import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from utils import print_step
from cache import is_in_cache, save_in_cache, load_cache


def run_fold_parallel(label, index, params, model_fn):
    index = index + 1
    print_step('Fold parallel {}/5'.format(index))
    if is_in_cache('{}_fold_preds_{}'.format(label, index)):
        print_step('Fold {} for {} already run!'.format(index, label))
        return None
    dev_X, val_X = load_cache('{}_fold_data_{}'.format(label, index), verbose=False)
    test, _ = load_cache('{}_fold_test_data_{}'.format(label, index), verbose=False)
    dev_y, val_y = load_cache('{}_fold_target_{}'.format(label, index), verbose=False)
    dev_y = dev_y['target']
    val_y = val_y['target']
    params2 = params.copy()
    pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
    save_in_cache('{}_fold_preds_{}'.format(label, index), pd.DataFrame({'pred': pred_val_y}), pd.DataFrame({'pred': pred_test_y}), verbose=False)
    if importances is not None:
        save_in_cache('{}_fold_importances_{}'.format(label, index), pd.DataFrame({'importance': importances}), None, verbose=False)
    print_step('Fold {} for {} complete!'.format(index, label))
    return None


def run_cv_model(train, test, target, model_fn, params, eval_fn, label='model', cores=5):
    kf = StratifiedKFold(n_splits=5, random_state=42)
    outlier_target = (target < -30).astype(int)
    fold_splits = kf.split(train, outlier_target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    feature_importance_df = pd.DataFrame()
    is_dataframe = isinstance(train, pd.DataFrame)
    train_cols = train.columns
    i = 1
    for dev_index, val_index in fold_splits:
        print_step('Started ' + label + ' fold ' + str(i) + '/5')
        if is_in_cache('{}_fold_data_{}'.format(label, i)):
            print_step('Fold {} for {} already has data!'.format(i, label))
        else:
            if isinstance(train, pd.DataFrame):
                dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
                dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
                dev_X = dev_X.reset_index().drop('index', axis=1)
                val_X = val_X.reset_index().drop('index', axis=1)
            else:
                dev_X, val_X = train[dev_index], train[val_index]
                dev_y, val_y = target[dev_index], target[val_index]
            dev_y = pd.DataFrame({'target': dev_y}).reset_index().drop('index', axis=1)
            val_y = pd.DataFrame({'target': val_y}).reset_index().drop('index', axis=1)
            save_in_cache('{}_fold_data_{}'.format(label, i), dev_X, val_X, verbose=False)
            save_in_cache('{}_fold_test_data_{}'.format(label, i), test, None, verbose=False)
            save_in_cache('{}_fold_target_{}'.format(label, i), dev_y, val_y, verbose=False)
            dev_index = pd.DataFrame({'index_label': dev_index}).reset_index().drop('index', axis=1)
            val_index = pd.DataFrame({'index_label': val_index}).reset_index().drop('index', axis=1)
            save_in_cache('{}_fold_index_{}'.format(label, i), dev_index, val_index, verbose=False)
            del dev_X
            del dev_y
            del val_X
            del val_y
            gc.collect()
        i += 1
    del train
    del test
    del dev_index
    del val_index
    del fold_splits
    del kf
    del outlier_target
    gc.collect()

    if cores > 1:
        print_step('NB: Running pool with {} cores'.format(cores))
        pool = mp.ProcessingPool(cores)
        pool.map(lambda dat: run_fold_parallel(dat[0], dat[1], dat[2], dat[3]),
                                               [(label, i, params, model_fn) for i in range(5)])
        pool.close()
        pool.join()
    else:
        print_step('NB: Parallel mode disabled.')
        list(map(lambda dat: run_fold_parallel(dat[0], dat[1], dat[2], dat[3]),
                                               [(label, i, params, model_fn) for i in range(5)]))

    for i in range(1, 6):
        print_step('Gathering ' + label + ' fold ' + str(i) + '/5')
        importances, _ = load_cache('{}_fold_importances_{}'.format(label, i), verbose=False)

        if importances is not None and is_dataframe:
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train_cols
            fold_importance_df['importance'] = importances['importance']
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        pred_val_y, pred_test_y = load_cache('{}_fold_preds_{}'.format(label, i), verbose=False)
        dev_y, val_y = load_cache('{}_fold_target_{}'.format(label, i), verbose=False)
        dev_index, val_index = load_cache('{}_fold_index_{}'.format(label, i), verbose=False)
        val_y = val_y['target']
        val_index = val_index['index_label']
        pred_val_y = pred_val_y['pred']
        pred_test_y = pred_test_y['pred']
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_scores.append(eval_fn(val_y, pred_val_y))

    print_step(label + ' cv scores : ' + str(cv_scores))
    print_step(label + ' mean cv score : ' + str(np.mean(cv_scores)))
    print_step(label + ' std cv score : ' + str(np.std(cv_scores)))
    final_cv = eval_fn(target, pred_train)
    print_step(label + ' final cv score : ' + str(final_cv))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
               'cv': cv_scores,
               'final_cv': final_cv,
               'importance': feature_importance_df}
    return results
