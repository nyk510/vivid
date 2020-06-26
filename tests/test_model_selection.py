import numpy as np
import pandas as pd
import pytest

from vivid.model_selection import ContinuousStratifiedFold
from vivid.model_selection import create_adversarial_dataset


@pytest.fixture
def continuous_y():
    y = [1, 1, 2, 2, 3, 3]
    y = np.asarray(y)
    return np.asarray(y)


def test_continuous_stratified(continuous_y):
    fold = ContinuousStratifiedFold(n_splits=2, shuffle=False, q=3)
    cv = list(fold.split(continuous_y, continuous_y))

    for idx_tr, idx_val in cv:
        for i in [1, 2, 3]:
            y_tr = continuous_y[idx_tr]
            assert (y_tr == i).sum() == 1


def test_raise_valueerror_on_continuous_stratified(continuous_y):
    # too many quantiles (20)
    fold = ContinuousStratifiedFold(n_splits=2, q=20)

    with pytest.raises(ValueError, match=r'Fail.*'):
        fold.split(continuous_y, continuous_y)


def test_make_adversarial_dataset():
    a = pd.DataFrame({
        'a': [1, 2, 4],
    })
    b = pd.DataFrame({
        'a': [2, 4, 5],
    })

    df, y = create_adversarial_dataset(a, b)

    for x in y[:len(a)]:
        assert x == 0

    for x in y[len(a):]:
        assert x == 1


def test_column_mismatch():
    a = pd.DataFrame({
        'a': [1, 3, 4]
    })

    b = pd.DataFrame({
        'b': [1, 3, 4]
    })

    with pytest.raises(ValueError):
        create_adversarial_dataset(a, b)


def test_type_mismatch():
    a = [[1, 3, 4]]
    b = [[1, 4, 5]]

    with pytest.raises(ValueError):
        create_adversarial_dataset(a, b)
