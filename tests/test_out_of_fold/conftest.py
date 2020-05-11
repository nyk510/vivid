import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_boston, load_breast_cancer

from vivid.env import Settings


@pytest.fixture(scope='function', autouse=True)
def stop_logging():
    before = Settings.LOG_LEVEL
    Settings.LOG_LEVEL = 'WARNING'
    yield
    Settings.LOG_LEVEL = before


@pytest.fixture
def regression_data() -> [pd.DataFrame, np.ndarray]:
    X, y = load_boston(True)
    return pd.DataFrame(X), y

@pytest.fixture
def binary_data() -> [pd.DataFrame, np.ndarray]:
    x, y = load_breast_cancer(True)
    return pd.DataFrame(x), y
