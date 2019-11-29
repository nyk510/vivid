import numpy as np
import pandas as pd


class DataLoader:
    _data = None

    def __init__(self, loader):
        """

        Args:
            loader:  callable object

        """
        self.loader = loader

    def _read_core(self):
        return self.loader()

    def read(self):
        if self._data is None:
            self._data = self._read_core()
        return self._data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        raise ValueError('data refer only. cant set')


def create_data_loader(loader):
    return DataLoader(loader)


def make_value_count_df(input_df: pd.DataFrame, whole_df: pd.DataFrame, c: str, dropna=False) -> pd.DataFrame:
    mapping = whole_df[c].value_counts(dropna=dropna).to_dict()
    df_out = pd.DataFrame(input_df[c].map(mapping))
    return df_out.add_prefix('count_')


def binning_value_count(input_df: pd.DataFrame, whole_df: pd.DataFrame, column: str,
                        bins=25, add_value=False):
    """
    binning したあとに value_count を行う

    Args:
        input_df:
        whole_df:
        column:
        bins:
        add_value: True の時 count にするまえのクラスIDも付与する

    Returns:

    """
    out_df = pd.DataFrame()

    counts, bins = np.histogram(whole_df[column].dropna(), bins=bins)
    c = pd.cut(input_df[column], bins=bins).cat.codes
    mapping = dict(c.value_counts())
    out_df[f'binning_{column}_count'] = c.map(mapping)  # code -> count の値に射影
    if add_value:
        out_df[f'binning_{column}_value'] = c
    return out_df
