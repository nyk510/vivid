import os
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_boston

from vivid.env import Settings
from vivid.fixture import cacheable, CacheFunctionFactory
from vivid.setup import setup_project


class CountLoader:
    def __init__(self, loader):
        self.loader = loader
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return self.loader(return_X_y=True)


@pytest.fixture(autouse=True)
def cache_dir(tmpdir):
    root = tmpdir
    Settings.PROJECT_ROOT = tmpdir
    if os.path.exists(root):
        shutil.rmtree(root)
    CacheFunctionFactory.wrappers = {}
    return setup_project(root).cache


@pytest.fixture(scope='function')
def boston_count_loader():
    return CountLoader(load_boston)


@pytest.mark.parametrize('inputs,expect', [
    (('foo', '/analysis/data', 'hoge'), '/analysis/data/foo.hoge'),
])
def test_get_cache_dir(inputs, expect):
    from vivid.fixture import _get_cache_dir

    assert _get_cache_dir(*inputs) == expect


def test_fixture_cache(boston_count_loader, cache_dir):
    @cacheable
    def boston():
        return boston_count_loader()

    X1, y1 = boston()
    assert isinstance(X1, np.ndarray)
    assert isinstance(y1, np.ndarray)

    assert boston_count_loader.counter == 1
    # by default, save to env.Settings.CACHE_DIR
    assert os.path.exists(os.path.join(cache_dir, 'boston.joblib'))

    X2, y2 = boston()
    assert isinstance(X2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    # create method is call at once
    assert boston_count_loader.counter == 1
    np.testing.assert_array_almost_equal(X1, X2)

    assert 'boston' in CacheFunctionFactory.list_keys()

    shutil.rmtree(cache_dir)

    boston()
    assert boston_count_loader.counter == 2
    boston()
    assert boston_count_loader.counter == 2

    CacheFunctionFactory.clear_cache()
    boston()
    assert boston_count_loader.counter == 3


def test_register_as_custom_name(boston_count_loader, cache_dir):
    @cacheable(callable_or_scope='other_boston')
    def boston():
        return boston_count_loader()

    X, y = boston()
    assert os.path.exists(os.path.join(cache_dir, 'other_boston.joblib'))


@pytest.mark.usefixtures('cache_dir')
@pytest.mark.parametrize('output', [
    pd.DataFrame([1, 2, 3, 2]),
    np.random.uniform(size=(100, 100)),
    {'hoge': [1, 2, 3]}
])
def test_generated_object_type(output):
    @cacheable
    def test():
        return output

    x1 = test()
    CacheFunctionFactory.clear_cache()
    x2 = test()


def test_list_key():
    @cacheable
    def hoge():
        return pd.DataFrame()

    @cacheable
    def foo():
        return pd.DataFrame()

    keys = CacheFunctionFactory.list_keys()
    assert 'foo' in keys
    assert 'hoge' in keys


def test_register_function():
    def create():
        return pd.DataFrame([1, 2, 3])

    wrapper = cacheable('hoge')(create)
    wrapper()
    assert 'hoge' in CacheFunctionFactory.list_keys()


def test_cant_register_not_callable():
    x = pd.DataFrame([1, 2, 3])
    with pytest.raises(ValueError):
        cacheable(x)

    cacheable(lambda: x.copy())


def test_register_same_name():
    def create_1():
        return pd.DataFrame([1, 2, 3])

    def create_2():
        return None

    cacheable('create')(create_1)

    with pytest.warns(UserWarning):
        cacheable('create')(create_2)

    assert len(CacheFunctionFactory.list_keys()) == 2, CacheFunctionFactory.list_keys()
