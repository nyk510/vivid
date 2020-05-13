import os
import shutil

import numpy as np
import pandas as pd
import pytest

from vivid.core import AbstractFeature

HOME = os.path.expanduser('~')
RECORDING_DIR = os.path.join(HOME, '.vivid', 'test')


class SampleFeature(AbstractFeature):
    def __init__(self):
        super(SampleFeature, self).__init__('sample')

    def call(self, df_source, y=None, test=False):
        return df_source


class RecordingFeature(AbstractFeature):
    def __init__(self):
        super(RecordingFeature, self).__init__(name='rec_sample', parent=None, root_dir=RECORDING_DIR)

    def call(self, df_source, y=None, test=False):
        return df_source


@pytest.fixture
def train_data() -> [pd.DataFrame, np.ndarray]:
    n_rows = 100
    n_cols = 10
    x = np.random.uniform(size=(n_rows, n_cols))
    y = np.random.uniform(size=(n_rows,))
    train_df = pd.DataFrame(x)
    return train_df, y


@pytest.fixture
def toy_df():
    x = [
        [1, 'foo'],
        [2, 'bar'],
        [5, 'poyo']
    ]
    return pd.DataFrame(data=x, columns=['int_type', 'string_type'])


@pytest.fixture
def output_dir() -> str:
    default_path = os.path.join(os.path.expanduser('~'), '.vivid', 'test_cache')
    return os.environ.get('OUTPUT_DIR', default_path)


@pytest.fixture(scope='function', autouse=True)
def clean_up(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    yield

    from vivid.env import Settings, get_dataframe_backend
    Settings.CACHE_ON_TRAIN = True
    Settings.CACHE_ON_TEST = True
    Settings.DATAFRAME_BACKEND = 'vivid.backends.dataframes.JoblibBackend'
    get_dataframe_backend.backend = None
    shutil.rmtree(output_dir)
    shutil.rmtree(RECORDING_DIR)


def test_sample_feature(train_data):
    train_df, y = train_data
    feat = SampleFeature()
    df = feat.fit(train_df, y=y)
    assert len(df) == len(train_data)
