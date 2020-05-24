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


def test_sample_feature(regression_set):
    train_df, y = regression_set
    feat = SampleFeature()
    df = feat.fit(train_df, y=y)
    assert len(df) == len(regression_set)
