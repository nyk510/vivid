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
    """Optimization class for indicators that need to be converted from continuous values to discrete values.

    It is assumed that the unique of the correct label `y_true` is int and continuous with a beginning of 0.
    Note that giving anything else is likely to return an incorrect threshold value
    """

    def __init__(self, objective, bins=None, minimize=True, method='Nelder-Mead'):
        """
        Args:
            objective:
                the objective function. The function should return a number
                with y_true and a discredited predicted value as input.
            bins:
                threshold values. This is the initial value of the optimization solver.
            minimize:
                treats it as a minimization problem when true
            method:
                optimization method. pass to scipy minimize.
        """
        self.bins = bins
        self.objective = objective
        self.minimize = minimize
        self.method = method
        self.result_ = None

    @property
    def optimized_bins(self):
        if self.result_ is None:
            return self.bins
        t = self.result_.get('x', None)
        return get_bins(t)

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        fit thresholds

        Args:
            y_true:
                target array.
            y_pred:
                predict array. it is the array as a continuous format

        Returns:
            optimized thresholds

        Examples:
            >>> from vivid.metrics import quadratic_weighted_kappa
            >>> optim = BinsOptimizer(quadratic_weighted_kappa, minimize=False)
            >>> y_pred = np.random.uniform(size=1000)
            >>> y_true = np.where(y_pred < .5, 0, 1)
            >>> y_pred = y_pred * .5
            >>> before = quadratic_weighted_kappa(y_true, np.round(y_pred))
            >>> before
            0.0
            >>> optim.fit(y_true, y_pred)
            [-inf, 0.24999999999999978, inf]
            >>> b = optim.predict(y_pred)
            >>> after = quadratic_weighted_kappa(y_true, b)
            >>> after
            1.0
        """

        def fnc(thresholds):
            bins = get_bins(thresholds)
            b_pred = pd.cut(y_pred, bins).codes
            loss = self.objective(y_true, b_pred)
            if not self.minimize:
                loss = - loss
            return loss

        if self.bins is None:
            self.bins = thresholds_from_y_true(y_true)
        self.result_ = minimize(fnc, self.bins, method=self.method)
        return self.optimized_bins

    def predict(self, y_continuous):
        return pd.cut(y_continuous, self.optimized_bins).codes
