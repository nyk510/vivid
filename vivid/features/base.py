from typing import Union, Iterable, List

import pandas as pd
from sklearn.base import BaseEstimator

from vivid.backends import ExperimentBackend
from vivid.core import BaseBlock


def get_target_columns(column: Union[str, List],
                       excludes: Union[None, List],
                       source_df: pd.DataFrame) -> Iterable:
    if column == '__all__':
        column = source_df.columns.tolist()

    if isinstance(column, str):
        return [column]
    if isinstance(column, Iterable):
        return [x for x in column]

    return []


class ColumnWiseBlock(BaseBlock):
    engine = None

    def __init__(self, name, column='__all__', excludes=None, **kwargs):
        super(ColumnWiseBlock, self).__init__(name=name, **kwargs)
        self.column = column
        self.excludes = excludes

    def create_new_engine(self, column_name: str) -> BaseEstimator:
        return self.engine()

    def unzip(self, experiment: ExperimentBackend):
        self.mappings_ = experiment.load_object('mapping')

    def frozen(self, experiment: ExperimentBackend):
        experiment.save_as_python_object('mapping', self.mappings_)

    def get_output_colname(self, column):
        return self.name + '_' + column

    def _fit_core(self, source_df, y, experiment) -> pd.DataFrame:
        columns = get_target_columns(self.column, excludes=self.excludes, source_df=source_df)

        mappings = {}
        for c in sorted(columns):
            clf = self.create_new_engine(c)
            clf.fit(source_df[c], y=y)
            mappings[c] = clf
        self.mappings_ = mappings
        return self.transform(source_df)

    def transform(self, source_df):
        out_df = pd.DataFrame()

        for column, clf in self.mappings_.items():
            out = clf.transform(source_df[column])
            out_df[self.get_output_colname(column)] = out
        return out_df
