import os

import numpy as np
import pandas as pd
import pytest
import shutil

from vivid.core import AbstractFeature
from .settings import OUTPUT_DIR
RECORDING_DIR = '/workspace/output'
os.makedirs(RECORDING_DIR, exist_ok=True)


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
def train_data():
    n_rows = 100
    n_cols = 10
    x = np.random.uniform(size=(n_rows, n_cols))
    y = np.random.uniform(size=(n_rows,))
    train_df = pd.DataFrame(x)
    return train_df, y


@pytest.fixture(scope='module')
def clean():
    shutil.rmtree(OUTPUT_DIR)
    yield


def test_sample_feature(train_data):
    train_df, y = train_data
    feat = SampleFeature()
    df = feat.fit(train_df, y=y)
    assert len(df) == len(train_data)
