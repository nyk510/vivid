# coding: utf-8
"""
モデル作成時に用いるクラスなどの定義
"""
from typing import Union

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from ..env import Settings
from ..utils import get_logger

__author__ = "nyk510"


def get_scalar_by_name(name):
    scaler = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    return scaler.get(name, None)


class UtilityTransform(TransformerMixin, BaseEstimator):
    """
    sklearn transformer にログスケール変換の処理を追加した transformer
    """

    def __init__(self,
                 log: bool = False,
                 scaling: Union[None, str] = None):
        super(UtilityTransform, self).__init__()
        self.log = log
        self.scaling = scaling
        self.threshold = 1e-10

    def fit(self, X, y=None):
        self.scalar_ = get_scalar_by_name(self.scaling)
        if self.scaling is not None and self.scalar_ is None:
            raise ValueError('')

        self.scalar_ = get_scalar_by_name(self.scaling)
        if self.log and np.sum(X < 0) > 0:
            raise ValueError('In Log-scalar, you must input value over zero')

        self.is_one_dim_ = len(X.shape) == 1
        if self.log:
            X = np.log1p(X + self.threshold)

        if self.scalar_:
            if self.is_one_dim_:
                X = X.reshape(-1, 1)
            X = self.scalar_.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_one_dim_')
        if self.log:
            X = np.log1p(X + self.threshold)

        if self.scalar_:
            if self.is_one_dim_:
                X = X.reshape(-1, 1)
            X = self.scalar_.transform(X)
            if self.is_one_dim_:
                X = X.reshape(-1, )
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, 'is_one_dim_')
        if self.scalar_:
            if self.is_one_dim_:
                X = X.reshape(-1, 1)
            X = self.scalar_.inverse_transform(X)

            if self.is_one_dim_:
                X = X.reshape(-1, )

        if self.log:
            X = np.expm1(X) - self.threshold
        return X


class PrePostProcessModel(BaseEstimator):
    """
    モデルの保存と入出力の正規化を行う機能を加えた scikit-learn estimator
    """

    def __init__(self,
                 instance: BaseEstimator,
                 input_scaling=None,
                 input_logscale=False,
                 target_scaling=None,
                 target_logscale=False,
                 verbose=1):
        """

        Args:
            model_class:
                学習させるモデルの class.
                model_class.fit を実装していること及び pickle で保存できる必要があります
            model_params(dict | None): モデルのパラメータ
            output_dir(str): モデルを保存するディレクトリのパス
            prepend_name(str): 最適なモデルを保存するときの prefix
            use_scaling(bool):
            num_cv_in_search:
            scoring(str): 探索時に使用する metrics
            logger:
        """

        self.input_scaling = input_logscale
        self.input_logscale = input_logscale
        self.target_scaling = target_logscale
        self.target_logscale = target_logscale

        self.input_transformer = UtilityTransform(input_logscale, input_scaling)
        self.target_transformer = UtilityTransform(target_logscale, target_scaling)
        self.verbose = verbose
        self.logger = get_logger(__name__, Settings.LOG_LEVEL)
        self.instance = instance
        super(PrePostProcessModel, self).__init__()

    def fit(self, x_train, y_train, **kwargs):
        clf = clone(self.instance)
        x, y = self._before_fit(x_train, y_train)
        # self.fit_params_ = kwargs
        self.fitted_model_ = clf.fit(x, y, **kwargs)
        return self

    def predict(self, x, prob=False):
        check_is_fitted(self, 'fitted_model_')
        x = self._before_predict(x)
        if prob:
            try:
                pred = self.fitted_model_.predict_proba(x)
            except AttributeError as e:
                self.logger.warning(e)
                pred = self.fitted_model_.predict(x)
        else:
            pred = self.fitted_model_.predict(x)

        pred = self.target_transformer.inverse_transform(pred)
        return pred

    def _before_fit(self, x_train, y_train):
        x = self.input_transformer.fit_transform(x_train)
        y = self.target_transformer.fit_transform(y_train)
        return x, y

    def _before_predict(self, x):
        """
        予測モデルに投入する前の前処理を行う

        Args:
            x: 変換前の特長

        Returns:
            変換後の特徴.
        """
        return self.input_transformer.transform(x)
