import numpy as np
import pytest
from sklearn.model_selection import KFold

from vivid.utils import get_train_valid_set, sigmoid


def test_get_train_test_set():
    fold1 = KFold(n_splits=4)
    x = [1, 2]
    y = [2, 3]

    with pytest.raises(ValueError):
        get_train_valid_set(fold1, x, y)

    y2 = []
    with pytest.raises(ValueError):
        get_train_valid_set(fold1, x, y2)


def test_sigmoid():
    x = 10000  # too large value
    y = sigmoid(-x)
    np.log(y)  # not set threshold, overflow in the log operation
