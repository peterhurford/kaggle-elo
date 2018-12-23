# Stratified fold strategy from https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
import numpy as np
import pandas as pd

import pathos.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from utils import print_step
from cache import get_data, is_in_cache, save_in_cache, load_cache


def run_cv_model(train, test, target, model_fn, params, eval_fn, label):
    kf = StratifiedKFold(n_splits=5, random_state=42)
    outlier_target = (target < -30).astype(int)
    fold_splits = kf.split(train, outlier_target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    feature_importance_df = pd.DataFrame()
    i = 1
    fold_data = []
    for dev_index, val_index in fold_splits:
        print_step('Started ' + label + ' fold ' + str(i) + '/5')
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
        save_in_cache('lgb_fold_data_{}'.format(i), dev_X, val_X)
        save_in_cache('lgb_fold_target_{}'.format(i), dev_y, val_y)
        i += 1

    def run_fold_parallel(index):
        index = index + 1
        print_step('Fold parallel {}/5'.format(index))
        dev_X, val_X = load_cache('lgb_fold_data_{}'.format(index))
        dev_y, val_y = load_cache('lgb_fold_target_{}'.format(index))
        dev_y = dev_y['target']
        val_y = val_y['target']
        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        save_in_cache('lgb_fold_preds_{}'.format(index), pd.DataFrame({'pred': pred_val_y}), pd.DataFrame({'pred': pred_test_y}))
        if importances is not None:
            save_in_cache('lgb_fold_importances_{}'.format(index), pd.DataFrame({'importance': importances}), None)
        return None

    n_cpu = mp.cpu_count()
    n_nodes = n_cpu - 3
    pool = mp.ProcessingPool(n_nodes)
    results = pool.map(run_fold_parallel, range(5))
    pool.close()
    pool.join()
    pool.restart()
    import pdb
    pdb.set_trace()

    if importances is not None and isinstance(train, pd.DataFrame):
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = train.columns
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    cv_score = eval_fn(val_y, pred_val_y)
    cv_scores.append(eval_fn(val_y, pred_val_y))
    print_step(label + ' cv score ' + str(i) + ' : ' + str(cv_score))
    i += 1
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
