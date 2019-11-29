import pytest
from sklearn.model_selection import KFold

from vivid.utils import get_train_valid_set


def test_get_train_test_set():
    fold1 = KFold(n_splits=4)
    x = [1, 2]
    y = [2, 3]

    with pytest.raises(ValueError):
        get_train_valid_set(fold1, x, y)

    y2 = []
    with pytest.raises(ValueError):
        get_train_valid_set(fold1, x, y2)
