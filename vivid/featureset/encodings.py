from typing import Type, List

import pandas as pd

from .atoms import AbstractAtom


class OneHotEncodingAtom(AbstractAtom):
    """use_columns に対して One Hot Encoding を実行する"""

    def __init__(self):
        super(OneHotEncodingAtom, self).__init__()
        self.master_df = None

    @property
    def is_fitted(self):
        return self.master_df is not None

    def call(self, df_input, y=None):
        out_df = pd.DataFrame()

        if not y is None:
            self.master_df = df_input.copy()

        for c in self.use_columns:
            x = df_input[c]
            cat = pd.Categorical(x, categories=sorted(self.master_df[c].unique()))
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

    def call(self, df_input, y=None):
        if y is not None:
            for c in self.use_columns:
                self.vc_set[c] = df_input[c].value_counts()

        out_df = pd.DataFrame()

        for k, v in self.vc_set.items():
            out_df[k] = df_input[k].map(v)

        return out_df.add_prefix('count_')


class InnerMergeAtom(AbstractAtom):
    """
    特定のカラムでの groupby 集計でカラムの値をその他のカラムの集計値に変換する
    """

    def __init__(self, merge_key, agg='mean'):
        self.agg = agg
        self.merge_key = merge_key
        self.merge_df = None
        super(InnerMergeAtom, self).__init__()

    @property
    def is_fitted(self):
        return self.merge_df is not None

    def call(self, df_input, y=None):
        value_columns = [c for c in self.use_columns if c != self.merge_key]
        columns = [self.merge_key, *value_columns]

        prefix = f'{self.merge_key}_{self.agg}_'

        if y is not None:
            _df = df_input.groupby(self.merge_key).aggregate(self.agg)[value_columns]
            _df = _df.add_prefix(prefix)
            self.merge_df = _df

        out_df = pd.merge(df_input[columns], self.merge_df, on=self.merge_key, how='left')

        for c in value_columns:
            out_df[f'{prefix}_diff_{c}'] = out_df[c] - out_df[f'{prefix}{c}']

        out_df = out_df.drop(columns=columns)
        return out_df


def make_inner_block(inner_merge_keys: List[str], merge_atom: Type[InnerMergeAtom]):
    atoms = []

    for i in inner_merge_keys:
        for agg in ['mean', 'max', 'std', 'min', 'median']:
            atoms.append(merge_atom(merge_key=i, agg=agg))

    return atoms
