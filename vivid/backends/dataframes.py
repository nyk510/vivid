import os
import warnings

import pandas as pd


class DataFrameBackend:
    ext = 'csv'

    def get_ext(self):
        ext = self.ext
        if '.' in self.ext:
            warnings.warn('`ext` not contains `"."`')
            ext = ext.replace('.', '')
        return ext

    def to_filename(self, path: str):
        name = os.path.basename(path)
        if '.' in name:
            name = name.split('.')[0]

        dirname = os.path.dirname(path)
        return os.path.join(dirname, f'{name}.{self.get_ext()}')

    def save(self, df: pd.DataFrame, save_to: str, **kwargs):
        raise NotImplementedError()

    def load(self, path_to_data: str, **kwargs):
        raise NotImplementedError()


class CSVBackend(DataFrameBackend):
    """save and load csv as text format"""

    def save(self, df: pd.DataFrame, save_to: str, index=False, **kwargs):
        df.to_csv(save_to, index=index, **kwargs)

    def load(self, path_to_data: str, **kwargs):
        return pd.read_csv(path_to_data, **kwargs)


class FeatherBackend(DataFrameBackend):
    """save and load feather-format library"""
    ext = 'feather'

    def save(self, df: pd.DataFrame, save_to: str, **kwargs):
        df.columns = df.columns.astype(str)
        df.to_feather(save_to)

    def load(self, path_to_data: str, **kwargs):
        import feather
        return feather.read_dataframe(path_to_data)


class JoblibBackend(DataFrameBackend):
    """save and load joblib"""
    ext = 'joblib'

    def save(self, df: pd.DataFrame, save_to: str, **kwargs):
        import joblib
        joblib.dump(df, save_to)

    def load(self, path_to_data: str, **kwargs):
        import joblib
        return joblib.load(path_to_data)
