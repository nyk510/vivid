"""define utility functions that calculate commonly use metrics
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, \
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


@dataclass
class Metric:
    name: str
    as_binary: False
    method: Callable


def as_binary(y_pred):
    return np.argmax(y_pred, axis=1)


class TopNAccuracyScore:
    def __init__(self, n=3):
        self.n = n

    def __call__(self, y_true, y_pred):
        top_n_labels = np.argsort(y_pred, axis=1)[:, -self.n:]
        same_labels = np.sum(top_n_labels == y_true.reshape(-1, 1), axis=1)
        return same_labels.sum() / len(same_labels)


def top_n_accuracy_score(y_true, y_pred, n=1) -> float:
    """
    Top nth accuracy score

    Args:
        y_true:
            true label. shape = (n_samples, )
        y_pred:
            predict probability. shape = (n_samples, n_classes).
        n:
            top n. must be gte 1.

    Returns:
        top `n` th accuracy score.
    """
    if n < 1:
        raise ValueError('`n` must be bigger than 1.')

    return TopNAccuracyScore(n=n)(y_true, y_pred)


REGRESSION_METRICS = {
    'rmsle': lambda *x: mean_squared_log_error(*x) ** .5,
    'rmse': root_mean_squared_error,
    'mean-ae': mean_absolute_error,
    'median-ae': median_absolute_error,
    'r2': r2_score,
    'explained_variance': explained_variance_score,
}


def binary_metrics(y_true: np.ndarray,
                   predict_probability: np.ndarray,
                   threshold=.5) -> dict:
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
        metrics metrics dictionary. the key is metric name, and the value is score.
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

    scores = {}
    for f in none_prob_functions:
        score = f(y_true, predict_label)
        scores[str(f.__name__)] = score
    for f in prob_functions:
        score = f(y_true, predict_probability)
        scores[f.__name__] = score

    return scores


def regression_metrics(y_true, y_pred) -> dict:
    """
    calculate regression task metrics

    Args:
        y_true:
            target. shape = (n_data,)
        y_pred:
            predict value. shape = (n_data,)

    Returns:
        regression metrics dictionary
    """

    scores = {}

    for k, func in REGRESSION_METRICS.items():
        try:
            scores[k] = func(y_true, y_pred)
        except Exception as e:
            scores[k] = None

    return scores


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_label=None) -> dict:
    """
    calculate multiclass task metrics.

    Args:
        y_true:
            target. shape = (n_samples, )
        y_pred:
            predict values (probability). shape = (n_samples, n_classes).
        pred_label:
            predict labels (Optional).
            If set None, create the labels using the maximum probability index by records.

    Returns:
        score dict.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must be same length')

    if pred_label is None:
        pred_label = as_binary(y_pred)

    metrics = [
        Metric(name='log_loss', as_binary=False, method=log_loss),
        Metric(name='accuracy_score', as_binary=True, method=accuracy_score),
        *[Metric(name='accuracy@{}'.format(n), as_binary=False,
                 method=TopNAccuracyScore(n=n)) for n in [3, 4, 5]]
    ]

    score = {}

    for m in metrics:
        pred_i = pred_label if m.as_binary else y_pred
        score[m.name] = m.method(y_true, pred_i)

    return score
