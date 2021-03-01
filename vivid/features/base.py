from typing import Union, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vivid.backends.experiments import ExperimentBackend
from vivid.core import BaseBlock
from .engine import BinCountEncoder, OneHotEncoder, CountEncoder, BaseEngine


def get_target_columns(source_df: pd.DataFrame,
                       column: Union[str, List] = '__all__',
                       excludes: Union[None, List] = None
                       ) -> Iterable:
    """
    select the target columns

    Args:
        source_df:
            input dataframe.
        column:
            Explicitly used columns. if set `"__all__"`, return all columns.
            column と exlucdes は同時に設定できません (どちらか一方だけ選択できます)
        excludes:
            Explicitly remove columns
    Returns:

    """

    use_all = column == '__all__'
    all_columns = source_df.columns.tolist()
    if isinstance(column, str):
        column = [column]

    if use_all:
        column = all_columns
    else:
        not_exist_columns = [c for c in column if c not in all_columns]
        if len(not_exist_columns) > 0:
            raise ValueError(
                'some specific column does not exist in source_df columns. i.e. {}'.format(','.join(not_exist_columns)))

    excludes = [] if excludes is None else excludes

    if len(excludes) == 0:
        return column

    # 以下 excludes が存在している場合.
    not_exist_columns = [c for c in excludes if c not in all_columns]
    if len(not_exist_columns) > 0:
        raise ValueError(
            'some specific `excludes` columns does not exist in source_df columns. i.e. {}'.format(','.join(not_exist_columns))
        )

    return [c for c in column if c not in excludes]


class ColumnWiseBlock(BaseBlock):
    """
    apply feature engineering for each columns.
    """
    engine = None

    def __init__(self,
                 name,
                 column: Union[str, List] = '__all__',
                 excludes: Union[None, List] = None,
                 **kwargs):
        """
        Args:
            name:
                this block name.
            column:
                use columns. if set `"__all__"`, use all columns get from parent blocks
            excludes:
                if set, exclude these columns from column.
                [NOTE]
                    when set specific column (ex. ['foo', 'bar']), excludes must be None.
            **kwargs:
        """
        super(ColumnWiseBlock, self).__init__(name=name, **kwargs)
        self.column = column
        self.excludes = excludes

    def create_new_engine(self, column_name: str) -> BaseEngine:
        return self.engine()

    def unzip(self, experiment: ExperimentBackend):
        self.fitted_models_ = experiment.load_object('mapping')
        return self

    def frozen(self, experiment: ExperimentBackend):
        experiment.save_as_python_object('mapping', self.fitted_models_)
        return self

    def get_output_colname(self, column):
        return self.name + '_' + column

    def fit(self, source_df, y, experiment) -> pd.DataFrame:
        columns = get_target_columns(source_df=source_df,
                                     column=self.column,
                                     excludes=self.excludes)

        mappings = {}
        for c in sorted(columns):
            clf = self.create_new_engine(c)
            clf.fit(source_df[c], y=y)
            mappings[c] = clf
        self.fitted_models_ = mappings
        return self.transform(source_df)

    def transform(self, source_df):
        out_df = pd.DataFrame()

        for column, clf in self.fitted_models_.items():
            out = clf.transform(source_df[column])
            out_df[self.get_output_colname(column)] = out
        return out_df


class FilterBlock(ColumnWiseBlock):
    def fit(self, source_df, y, experiment) -> pd.DataFrame:
        return self.transform(source_df)

    def transform(self, source_df):
        cols = get_target_columns(source_df, column=self.column, excludes=self.excludes)
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

        for column, clf in self.fitted_models_.items():  # type: (str, OneHotEncoder)
            out_i = clf.transform(source_df[column].values)
            categories = clf.cats_
            _df = pd.DataFrame(out_i, columns=[str(c) for c in categories])
            _df = _df.add_prefix('{}='.format(column))
            out_df = pd.concat(
                [out_df, _df]
                , axis=1)
        return out_df


class CountEncodingBlock(ColumnWiseBlock):
    engine = CountEncoder

    def get_output_colname(self, column):
        return 'CE_{}'.format(column)


class FillnaBlock(BaseBlock):
    _save_attributes = [
        'fill_values_'
    ]

    def fit(self,
            source_df: pd.DataFrame,
            y: Union[None, np.ndarray],
            experiment: ExperimentBackend) -> pd.DataFrame:
        self.fill_values_ = source_df.median()
        return self.transform(source_df)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        out_df = source_df.fillna(self.fill_values_)
        return out_df
