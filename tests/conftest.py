import os
import shutil

import numpy as np
import pandas as pd
import pytest

from vivid.core import BaseBlock

HOME = os.path.expanduser('~')
RECORDING_DIR = os.path.join(HOME, '.vivid', 'test')


class SampleFeature(BaseBlock):
    def __init__(self):
        super(SampleFeature, self).__init__('sample')

    def transform(self, source_df):
        return source_df


class CounterBlock(BaseBlock):
    def __init__(self, **kwargs):
        super(CounterBlock, self).__init__(**kwargs)
        self.counter = 0

    def fit(self, source_df, y, experiment) -> pd.DataFrame:
        self.counter += 1
        return self.transform(source_df)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        out = source_df.copy()
        return out.add_prefix(self.name)


@pytest.fixture
def regression_Xy():
    return np.random.uniform(size=(100, 20)), np.random.uniform(size=(100,))


@pytest.fixture
def binary_Xy(regression_Xy):
    X, y = regression_Xy
    return X, np.where(y > .5, 1, 0)


@pytest.fixture
def regression_set(regression_Xy) -> [pd.DataFrame, np.ndarray]:
    return pd.DataFrame(regression_Xy[0]), regression_Xy[1]


@pytest.fixture
def regression_hasna_set() -> [pd.DataFrame, np.ndarray]:
    N = 1000
    dim = 10

    x = np.random.uniform(0, 1, size=(N, dim))
    x = np.where(x < .5, np.nan, x)
    y = np.random.uniform(size=N)

    df = pd.DataFrame(x)

    return df, y


@pytest.fixture
def toy_df() -> pd.DataFrame:
    data = {
        'int_type': [1, 2, 3, 2, 1],
        'string_type': ['hoge', 'hoge', 'foo', 'foo', 'bar']
    }
    return pd.DataFrame(data)


@pytest.fixture
def output_dir() -> str:
    default_path = os.path.join(os.path.expanduser('~'), '.vivid', 'test_cache')
    path = os.environ.get('OUTPUT_DIR', default_path)
    return path


@pytest.fixture(scope='function', autouse=True)
def clean_up(tmpdir: str):
    os.makedirs(tmpdir, exist_ok=True)
    yield

    from vivid.env import Settings, get_dataframe_backend
    Settings.CACHE_ON_TRAIN = True
    Settings.CACHE_ON_TEST = True
    Settings.DATAFRAME_BACKEND = 'vivid.backends.dataframes.JoblibBackend'
    get_dataframe_backend.backend = None
    shutil.rmtree(tmpdir)
