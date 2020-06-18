import numpy as np
import pytest

from vivid.model_selection import ContinuousStratifiedFold


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
