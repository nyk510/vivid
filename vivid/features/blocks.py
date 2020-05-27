import pandas as pd
from sklearn.base import BaseEstimator

from vivid.backends.experiments import ExperimentBackend
from .base import ColumnWiseBlock, get_target_columns
from .engine import BinCountEncoder, OneHotEncoder, CountEncoder


class FilterBlock(ColumnWiseBlock):
    def fit(self, source_df, y, experiment) -> pd.DataFrame:
        return self.transform(source_df)

    def transform(self, source_df):
        cols = get_target_columns(self.column, self.excludes, source_df)
        return source_df[cols].copy()

    def frozen(self, experiment: ExperimentBackend):
        return self

    def unzip(self, experiment: ExperimentBackend):
        return self


class BinningCountBlock(ColumnWiseBlock):
    engine = BinCountEncoder

    def __init__(self, name, column='__all__', bins=25, **kwargs):
        super(BinningCountBlock, self).__init__(name=name, column=column, **kwargs)
        self.bins = bins

    def create_new_engine(self, column_name: str):
        return self.engine(bins=self.bins)

    def get_output_colname(self, column):
        return '{}_bins={}'.format(column, self.bins)


class OneHotEncodingBlock(ColumnWiseBlock):
    engine = OneHotEncoder

    def create_new_engine(self, column_name: str) -> BaseEstimator:
        return self.engine(min_freq=0, max_columns=20)

    def transform(self, source_df):
        out_df = pd.DataFrame()

        for column, clf in self.mappings_.items():
            out_i = clf.transform(source_df[column].values)
            out_df = pd.concat(
                [out_df, pd.DataFrame(out_i, columns=['{}_{}'.format(column, i) for i in range(len(out_i.T))])]
                , axis=1)
        return out_df


class CountEncodingBlock(ColumnWiseBlock):
    engine = CountEncoder

    def get_output_colname(self, column):
        return 'CE_{}'.format(column)
