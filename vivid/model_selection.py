import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import _deprecate_positional_args


class StratifiedGroupKFold(StratifiedKFold):
    """
    Stratified Group-K-Fold Splits
    """

    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, q=10, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
        self.q = q

    def split(self, X, y, groups=None):
        y_cat = pd.qcut(y, q=self.q).codes
        return super(StratifiedGroupKFold, self).split(X, y_cat, groups=groups)
