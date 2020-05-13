# coding: utf-8
"""
モデル作成時に用いるクラスなどの定義
"""
import os

import joblib
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..env import Settings
from ..utils import get_logger

__author__ = "nyk510"


def get_scalar_by_name(name):
    scaler = {
        'standard': StandardScaler()
    }
    return scaler.get(name, None)


class UtilityTransform(BaseEstimator, TransformerMixin):
    """
    sklearn transformer にログスケール変換の処理を追加した transformer
    """

    def __init__(self, log=False, scaling=None):
        self.log = log
        self.scaling = get_scalar_by_name(scaling)
        self.threshold = 1e-10

    @property
    def use_scaling(self):
        return self.scaling is not None

    def fit(self, x, y=None):
        if self.log and np.sum(x < 0) > 0:
            raise ValueError('In Logscalar, you must input value over zero')

        self.is_one_dim_ = len(x.shape) == 1
        if self.log:
            x = np.log1p(x + self.threshold)

        if self.use_scaling:
            if self.is_one_dim_:
                x = x.reshape(-1, 1)
            x = self.scaling.fit_transform(x)
        return self

    def transform(self, x):
        check_is_fitted(self, 'is_one_dim_')
        if self.log:
            x = np.log1p(x + self.threshold)

        if self.use_scaling:
            if self.is_one_dim_:
                x = x.reshape(-1, 1)
            x = self.scaling.transform(x)
            if self.is_one_dim_:
                x = x.reshape(-1, )
        return x

    def inverse_transform(self, x):
        check_is_fitted(self, 'is_one_dim_')
        if self.use_scaling:
            if self.is_one_dim_:
                x = x.reshape(-1, 1)
            x = self.scaling.inverse_transform(x)

            if self.is_one_dim_:
                x = x.reshape(-1, )

        if self.log:
            x = np.expm1(x) - self.threshold
        return x


class PrePostProcessModel(BaseEstimator):
    """
    モデルの保存と入出力の正規化を行う機能を加えた scikit-learn estimator
    """

    def __init__(self,
                 model_class,
                 model_params=None,
                 input_scaling=None,
                 input_logscale=False,
                 target_scaling=None,
                 target_logscale=False,
                 output_dir=None,
                 verbose=1,
                 logger=None):
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
        self.model_class = model_class

        if isinstance(model_params, dict):
            self.model_params = model_params
        elif model_params is None:
            self.model_params = {}
        else:
            raise ValueError('`model_param` must be dict or None. actually: {}'.format(model_params))

        self.input_transformer = UtilityTransform(input_logscale, input_scaling)
        self.target_transformer = UtilityTransform(target_logscale, target_scaling)

        self.output_dir = output_dir
        self.verbose = verbose

        if logger is None:
            self.logger = get_logger(__name__, Settings.LOG_LEVEL)
        else:
            self.logger = logger

    @property
    def is_recording(self):
        return self.output_dir is not None

    @property
    def model_path(self):
        if not self.is_recording:
            return None
        return os.path.join(self.output_dir, 'best_fitted.joblib')

    def create_model(self):
        self.logger.debug('Model Params')

        for k, v in self.model_params.items():
            self.logger.debug(f'{k}\t{v}')

        return self.model_class(**self.model_params)

    def load_trained_model(self):
        """
        :return: self
        :rtype: PrePostProcessModel
        """
        self.logger.debug('load model: {}'.format(self.model_path))
        self.fitted_model_ = joblib.load(self.model_path)

        self.input_transformer = joblib.load(os.path.join(self.output_dir, 'input.joblib'))
        self.target_transformer = joblib.load(os.path.join(self.output_dir, 'target.joblib'))

        return self

    def save_trained_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info('save to: {}'.format(self.model_path))
        joblib.dump(self.fitted_model_, self.model_path)
        joblib.dump(self.target_transformer, os.path.join(self.output_dir, 'target.joblib'))
        joblib.dump(self.input_transformer, os.path.join(self.output_dir, 'input.joblib'))

    def fit(self, x_train, y_train, **kwargs):
        """
        学習を実行するメソッド
        与えられたトレーニングデータで学習を行った後, 学習済みモデルを保存する

        Args:
            x_train(np.ndarray):
            y_train(np.ndarray):
            valid_set:
            fit_params(dict):

        Returns: fitted model instance

        """
        clf = self.create_model()
        x, y = self._before_fit(x_train, y_train)
        self.fit_params_ = kwargs
        self.fitted_model_ = clf.fit(x, y, **kwargs)
        if self.is_recording:
            self.save_trained_model()
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
