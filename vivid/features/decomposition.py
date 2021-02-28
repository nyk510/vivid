from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from vivid.backends.experiments import ExperimentBackend
from vivid.core import BaseBlock
from vivid.features.base import get_target_columns


class BaseDecompositionBlock(BaseBlock):
    model_params = {}
    model_class = None
    prefix = ''
    _save_attributes = [
        'clf_',
        'use_columns_'
    ]

    def __init__(self,
                 name: str,
                 columns='__all__',
                 add_fit_params: Union[None, dict] = None,
                 **kwargs):
        super(BaseDecompositionBlock, self).__init__(name=name, **kwargs)

        self.columns = columns

        if add_fit_params is None: add_fit_params = {}
        self.add_fit_params = add_fit_params

        self.clf_ = None
        self.use_columns_ = None

    def get_prefix(self):
        return self.prefix

    def _get_fit_params(self):
        params = deepcopy(self.model_params)
        params.update(self.add_fit_params)
        return params

    def get_model_instance(self) -> BaseEstimator:
        if self.model_class is None:
            raise NotImplementedError()

        return self.model_class(**self._get_fit_params())

    def fit(self,
            source_df: pd.DataFrame,
            y: Union[None, np.ndarray],
            experiment: ExperimentBackend) -> pd.DataFrame:
        columns = get_target_columns(source_df, column=self.columns)
        clf = self.get_model_instance()
        clf.fit(source_df[columns].values, y=y)
        self.clf_ = clf
        self.use_columns_ = columns
        return self.transform(source_df)

    def _predict(self, estimator, X) -> np.ndarray:
        if isinstance(estimator, TransformerMixin):
            return estimator.transform(X)

        if is_classifier(estimator):
            return estimator.predict_proba(X)

        return estimator.predict(X)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        X = source_df[self.use_columns_].values
        z = self._predict(self.clf_, X)
        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.get_prefix()}_')


class PCABlock(BaseDecompositionBlock):
    prefix = 'PCA_'
    model_class = PCA
    model_params = {
        'n_components': 2
    }


class GaussianMixtureBlock(BaseDecompositionBlock):
    prefix = 'GMM_'
    model_class = GaussianMixture
    model_params = {
        'n_components': 2,
    }
