# coding: utf-8
"""よく使う評価指標を計算する関数などを定義する
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, \
    r2_score, mean_squared_log_error, median_absolute_error, explained_variance_score, cohen_kappa_score, \
    average_precision_score, precision_score, recall_score

__author__ = "nyk510"


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average', squared=True):
    return mean_squared_error(y_true, y_pred,
                              sample_weight=sample_weight,
                              multioutput=multioutput, squared=squared) ** .5


def binary_metrics(y_true, predict_probability, threshold=.5) -> pd.DataFrame:
    """
    二値分類でよく使う評価指標全部入りの DataFrame を作成するメソッド
    Args:
        y_true(np.ndarray): 正解ラベルの配列. shape = (n_samples, )
        predict_probability(np.ndarray): 予測値の確率が入った配列. shape = (n_samples, )
        threshold(float): この値を超えたものをラベル 1 を予測したとみなす

    Returns:
        pd.DataFrame

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
    name_func_map = OrderedDict({
        'rmse': lambda *x: mean_squared_error(*x) ** .5
    })

    for m in [mean_squared_log_error, median_absolute_error, median_absolute_error, mean_squared_error, r2_score,
              explained_variance_score, mean_absolute_error]:
        name_func_map[m.__name__] = m

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


def upper_accuracy(y_true, predict, n_cut=20):
    """
    upper accuracy を算出する

    upper accuracy とは予測値をある閾値 `threshold` で区切り, 上位集合のうち
    実際に `label = 1` のものがどの程度存在しているかを計算したものです。
    予測値が大きいほど実際にラベルも1になってほしいので, 横軸にしきい値縦軸にその集合での label = 1 の割合 `accuracy` をプロットしたときに
    単調増加する予測は良い予測と言えます.

    Args:
        y_true(np.ndarray): shape = (n_samples,)
        predict:

    Returns:
        pd.DataFrame:
            しきい値ごとの正答率のデータ.
            カラムは `['threshold', 'count', 'hist', 'accuracy', 'ratio']` の順
            * threshold: predict を区切った際のしきい値
            * count: しきい値を超えたデータ点の数. int
            * hit: しきい値を超えたデータのうち実際に y_true = 1 が満たされた点の数. int
            * accuracy: しきい値を超えたデータのうち y_true = 1 の割合. hit / count と同値
            * ratio: しきい値が全体の何割で区切られたかを表す数値. float
    """
    df_metric = pd.DataFrame([predict, y_true], index=['predict', 'label']).T
    predict = df_metric['predict']
    data = []
    for r in np.linspace(0, 1, n_cut + 1)[:-1]:
        threshold = predict.quantile(r)
        df_i = df_metric[predict > threshold]
        count = len(df_i)
        hit = df_i['label'].sum()
        acc = hit / count
        data.append([threshold, count, hit, acc, r])

    df_upper_acc = pd.DataFrame(data, columns=['threshold', 'count', 'hist', 'accuracy', 'ratio'])
    return df_upper_acc
