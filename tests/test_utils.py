import numpy as np
import pytest
from sklearn.model_selection import KFold

from vivid.utils import get_train_valid_set
from vivid.utils import param_to_name
from vivid.utils import sigmoid


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


def test_p2n_reproduction_of_param_to_name():
    """parameter の順番が違っていても同じ name が作成されることを確認するテスト"""
    param1 = {
        'foo': 'bar',
        'hoge': 1
    }

    name1 = param_to_name(param1)

    # 順番が違う
    param2 = {
        'hoge': 1,
        'foo': 'bar'
    }

    name2 = param_to_name(param2)
    assert name1 == name2


def test_p2n_none_string_values():
    s = param_to_name({
        'foo': np.array([10, 20])
    })
    assert s == 'foo=[10 20]'


def test_p2n_change_separator():
    s = param_to_name({'foo': 'bar', 'a': 'b'}, key_sep='__', key_value_sep='~')
    assert s == 'a~b__foo~bar'
