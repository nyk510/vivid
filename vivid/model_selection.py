from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import _deprecate_positional_args


class ContinuousStratifiedFold(StratifiedKFold):
    """
    stratified-K-Fold Splits for continuous target.
    """

    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, q=10, shuffle=False, random_state=None):
        """
        Args:
            n_splits:
                number of splits
            q:
                number of quantiles.
                例えば10に設定されると, `y` を値に応じて 10 個の集合に分割し, 各 fold では train / valid に集合ごとの
                割合が同じように選択されます.
            shuffle:
                True の時系列をシャッフルしてから分割します
            random_state:
                シャッフル時のシード値

        Notes:
            y の水準数 (i.e. uniqueな数値の数) よりも大きい値を設定すると集合の分割に失敗し split を実行する際に
            ValueError となることがあります. その場合小さい `q` を設定することを検討してください
        """
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
        self.q = q

    def split(self, X, y, groups=None):
        try:
            y_cat = pd.qcut(y, q=self.q).codes
        except ValueError as e:
            raise ValueError('Fail to quantile cutting (using pandas.qcut). '
                             'There are cases where this value fails when you make it too large') from e
        return super(ContinuousStratifiedFold, self).split(X, y_cat, groups=groups)


def create_adversarial_dataset(train_df: pd.DataFrame,
                               test_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    create train / target tuple for adversarial validation.
    The target to be created is an array with 0 for the training data and 1 for the test data.

    Args:
        train_df:
            training data feature
        test_df:
            test data feature

    Returns:
        (feature, target) set.
    """
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError('train data must be pandas.DataFrame')
    if not isinstance(test_df, pd.DataFrame):
        raise ValueError('test data must be pandas.DataFrame')

    tr_cols = set(train_df.keys())
    test_cols = set(test_df.keys())

    inter = tr_cols and test_cols
    only_tr = tr_cols - inter
    only_test = test_cols - inter

    if only_tr:
        raise ValueError('the following columns exist only train. {}'.format(','.join(only_tr)))
    if only_test:
        raise ValueError('the following columns exist only test. {}'.format(','.join(only_test)))

    y = [0] * len(train_df) + [1] * len(test_df)
    y = np.array(y)
    out_df = pd.concat([train_df, test_df], sort=True, ignore_index=True)
    out_df = out_df[train_df.columns]

    return out_df, y
