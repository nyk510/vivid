"""define utility functions that calculate commonly use metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, \
    r2_score, mean_squared_log_error, median_absolute_error, explained_variance_score, cohen_kappa_score, \
    average_precision_score, precision_score, recall_score


def quadratic_weighted_kappa(y_true, y_pred):
    """
    QWK (Quadratic Weighted Kappa) Score

    Args:
        y_true:
            target array.
        y_pred:
            predict array. must be a discrete format.

    Returns:
        QWK score
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average', squared=True):
    return mean_squared_error(y_true, y_pred,
                              sample_weight=sample_weight,
                              multioutput=multioutput, squared=squared) ** .5


def binary_metrics(y_true: np.ndarray, predict_probability: np.ndarray, threshold=.5) -> pd.DataFrame:
    """
    calculate binary task metrics

    Args:
        y_true:
            target. shape = (n_data,)
        predict_probability:
            predict value. be probability prediction for log_loss, roc_auc, etc.
        threshold:
            Thresholds for calculating the metrics that need to be evaluated as labels
    Returns:
        metrics dataframe
    """
    predict_label = np.where(predict_probability > threshold, 1, 0)
    none_prob_functions = [
        accuracy_score,
        f1_score,
        precision_score,
        recall_score
    ]

    prob_functions = [
        roc_auc_score,
        log_loss,
        average_precision_score
    ]

    scores, indexes = [], []
    for f in none_prob_functions:
        score = f(y_true, predict_label)
        scores.append(score)
        indexes.append(f.__name__)
    for f in prob_functions:
        score = f(y_true, predict_probability)
        scores.append(score)
        indexes.append(f.__name__)

    return pd.DataFrame(data=scores, index=indexes, columns=['score'])


def regression_metrics(y_true, predict) -> pd.DataFrame:
    """
    calculate regression task metrics

    Args:
        y_true:
            target. shape = (n_data,)
        predict:
            predict value. shape = (n_data,)

    Returns:
        regression metrics dataframe
    """
    name_func_map = {
        'rmsle': lambda *x: mean_squared_log_error(*x) ** .5,
        'rmse': root_mean_squared_error,
        'mean-ae': mean_absolute_error,
        'median-ae': median_absolute_error,
        'r2': r2_score,
        'explained_variance': explained_variance_score,
    }

    data, idx = [], []

    for name in name_func_map.keys():
        idx.append(name)

        try:
            f = name_func_map[name]
            p = np.copy(predict)

            if f == mean_squared_log_error:
                p[p < .0] = 0.

            m = name_func_map[name](y_true, p)
        except Exception:
            m = None
        data.append(m)

    df_metrics = pd.DataFrame(data, index=idx, columns=['score'])
    return df_metrics
