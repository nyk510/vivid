from collections import OrderedDict
from typing import Type, List

import pandas as pd

from .atoms import AbstractAtom


class OneHotEncodingAtom(AbstractAtom):
    """use_columns に対して One Hot Encoding を実行する"""

    def __init__(self):
        super(OneHotEncodingAtom, self).__init__()
        self.mapping_ = None

    @property
    def is_fitted(self):
        return self.mapping_ is not None

    def fit(self, input_df: pd.DataFrame, y=None):
        self.mapping_ = OrderedDict()
        for c in self.use_columns:
            cat = sorted(input_df[c].dropna().unique())
            self.mapping_[c] = cat
        return self

    def transform(self, input_df):
        out_df = pd.DataFrame()

        for c in self.use_columns:
            x = input_df[c]
            cat = pd.Categorical(x, categories=self.mapping_[c])
            df_i = pd.get_dummies(cat, prefix=f'{c}_')
            df_i.columns = list(df_i.columns)
            out_df = pd.concat([out_df, df_i], axis=1)

        return out_df


class CountEncodingAtom(AbstractAtom):
    """Training Data を master set とみなしCount Encoding を実行する"""

    def __init__(self):
        super(CountEncodingAtom, self).__init__()
        self.vc_set = {}

    @property
    def is_fitted(self):
        return len(self.vc_set) > 0

    def fit(self, input_df: pd.DataFrame, y=None):
        for c in self.use_columns:
            self.vc_set[c] = input_df[c].value_counts()
        return self

    def transform(self, input_df):
        out_df = pd.DataFrame()

        for k, v in self.vc_set.items():
            out_df[k] = input_df[k].map(v)

        return out_df.add_prefix('count_')


class InnerMergeAtom(AbstractAtom):
    """
    特定のカラムでの groupby 集計でカラムの値をその他のカラムの集計値に変換する
    """

    def __init__(self, merge_key, agg='mean'):
        self.agg = agg
        self.merge_key = merge_key
        self.prefix = f'{self.merge_key}_{self.agg}_'
        self.merge_df = None
        super(InnerMergeAtom, self).__init__()

    @property
    def is_fitted(self):
        return self.merge_df is not None

    @property
    def value_columns(self):
        return [c for c in self.use_columns if c != self.merge_key]

    def fit(self, input_df: pd.DataFrame, y=None):
        if y is not None:
            _df = input_df.groupby(self.merge_key).aggregate(self.agg)[self.value_columns]
            _df = _df.add_prefix(self.prefix)
            self.merge_df = _df
        return self

    def transform(self, input_df):
        out_df = pd.merge(input_df[self.use_columns], self.merge_df, on=self.merge_key, how='left')
        add_df = self._add_math_operation(out_df)
        out_df = pd.concat([out_df, add_df], axis=1)
        out_df = out_df.drop(columns=self.use_columns)
        return out_df

    def _add_math_operation(self, input_df):
        out_df = pd.DataFrame()

        for c in self.value_columns:
            out_df[f'{self.prefix}_diff_{c}'] = input_df[c] - input_df[f'{self.prefix}{c}']
        return out_df


def make_inner_block(inner_merge_keys: List[str], merge_atom: Type[InnerMergeAtom]):
    atoms = []

    for i in inner_merge_keys:
        for agg in ['mean', 'max', 'std', 'min', 'median', 'nunique']:
            atoms.append(merge_atom(merge_key=i, agg=agg))

    return atoms
