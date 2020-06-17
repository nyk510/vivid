from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class BaseEngine(TransformerMixin, BaseEstimator):
    """単に明示的に fit / transform を定義している scikit-learn 準拠の transformer"""

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()


class OneHotEncoder(BaseEngine):
    """use_columns に対して One Hot Encoding を実行する"""

    def __init__(self,
                 min_freq: Union[int, float] = 0,
                 max_columns: Union[None, int, float] = None):
        self.mapping_ = None
        self.min_freq = min_freq
        self.max_columns = max_columns
        super(OneHotEncoder, self).__init__()

    @property
    def is_fitted(self):
        return self.mapping_ is not None

    def fit(self, X, y=None):
        vc = pd.Series(X).dropna().value_counts()

        if 0 < self.min_freq < 1.:
            min_count = len(X) * self.min_freq
        elif self.min_freq <= 0:
            min_count = 0
        else:
            min_count = int(self.min_freq)

        cats = vc[vc >= min_count].index
        if self.max_columns is not None and len(cats) > self.max_columns:
            n = max(0, self.max_columns)
            n = np.floor(n)
            cats = cats[:int(n)]
        self.cats_ = cats
        return self

    def transform(self, X):
        cat = pd.Categorical(X, categories=self.cats_)
        df_i = pd.get_dummies(cat, dummy_na=False)
        return df_i.values


class BinCountEncoder(BaseEngine):
    def __init__(self, bins=25):
        self.bins = bins

    def fit(self, X, y=None):
        x = pd.Series(X)
        counts, bins = np.histogram(x.dropna(), bins=self.bins)
        codes = pd.cut(x, bins=bins).cat.codes
        mapping = dict(codes.value_counts())
        self.mapping_ = mapping
        self.codes_ = codes
        self.actual_bins_ = bins
        return self

    def transform(self, X):
        c = pd.cut(X, bins=self.actual_bins_).cat.codes
        x = c.map(self.mapping_)
        return x


class CountEncoder(BaseEngine):
    """Training Data を master set とみなしCount Encoding を実行する"""

    def __init__(self):
        super(CountEncoder, self).__init__()

    def fit(self, X, y=None):
        self.vc_ = pd.Series(X).value_counts(dropna=False)
        return self

    def transform(self, X):
        return pd.Series(X).map(self.vc_).values
