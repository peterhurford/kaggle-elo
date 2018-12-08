import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from utils import print_step


def run_cv_model(train, test, target, model_fn, params, eval_fn, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=2017)
    fold_splits = kf.split(train)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print_step('Started ' + label + ' fold ' + str(i) + '/5')
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]

        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        if importances is not None:
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
