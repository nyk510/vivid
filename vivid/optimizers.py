import numpy as np
import pandas as pd
from scipy.optimize import minimize


def thresholds_from_y_true(y_true):
    y_unique = np.unique(y_true)
    y_thresholds = (y_unique[1:] + y_unique[:-1]) / 2
    return y_thresholds


def get_bins(thresholds):
    thresholds = np.sort(thresholds)
    return [-np.inf, *thresholds, np.inf]


class BinsOptimizer:
    """しきい値を最適化する class

    正解ラベル `y_true`の unique は 0 始まりのintかつ連続であると想定しています.
    それ以外のものを与えると間違ったしきい値を返す可能性が高いため注意して下さい
    """

    def __init__(self, objective, bins=None, minimize=True):
        """

        Args:
            objective: 目的関数
            bins: default のしきい値
            minimize: True の時最小化問題として取り扱います
        """
        self.bins = bins
        self.objective = objective
        self.minimize = minimize
        self.result_ = None

    @property
    def optimized_bins(self):
        if self.result_ is None:
            return self.bins
        t = self.result_.get('x', None)
        return get_bins(t)

    def fit(self, y_true, y_pred):
        def fnc(thresholds):
            bins = get_bins(thresholds)
            b_pred = pd.cut(y_pred, bins).codes
            loss = self.objective(y_true, b_pred)
            if not self.minimize:
                loss = - loss
            return loss

        if self.bins is None:
            self.bins = thresholds_from_y_true(y_true)
        self.result_ = minimize(fnc, self.bins, method='Nelder-Mead')
        return self.optimized_bins

    def predict(self, y_continuous):
        return pd.cut(y_continuous, self.optimized_bins).codes
