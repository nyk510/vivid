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
