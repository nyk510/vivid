import pandas as pd

from vivid.backends.experiments import ExperimentBackend
from .base import ColumnWiseBlock, get_target_columns
from .engine import BinCountEncoder


class FilterBlock(ColumnWiseBlock):
    def _fit_core(self, source_df, y, experiment) -> pd.DataFrame:
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
