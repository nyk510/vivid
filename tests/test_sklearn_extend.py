"""
sklearn extend のテストコード
"""
import os

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeClassifierCV
from sklearn.utils.validation import NotFittedError
from xgboost import XGBClassifier

from vivid.sklearn_extend import UtilityTransform, PrePostProcessModel
from .utils import is_close_to_zero


@pytest.mark.parametrize('scaling', [None, 'standard', 'minmax'])
@pytest.mark.parametrize('log', [True, False])
def test_basic(log, scaling):
    transformer = UtilityTransform()

    x = np.random.uniform(size=100)
    x_edited = transformer.fit_transform(x)
    assert is_close_to_zero(x, x_edited)

    transformer = UtilityTransform(log=log, scaling=scaling)
    x_trans = transformer.fit_transform(x)
    x_inv = transformer.inverse_transform(x_trans)
    assert is_close_to_zero(x, x_inv)


def test_out_of_scaling_string(regression_Xy):
    transformer = UtilityTransform(scaling='hogehoge')
    assert transformer.scaling == 'hogehoge'

    with pytest.raises(ValueError):
        transformer.fit(*regression_Xy)


def test_raise_value_error_logscale():
    transformer = UtilityTransform(log=True)
    X = np.random.uniform(-2, -1, size=100)
    with pytest.raises(ValueError):
        transformer.fit(X, y=None)


def test_not_fitteed_error():
    transformer = UtilityTransform()
    x = np.random.uniform(size=10)
    with pytest.raises(NotFittedError):
        transformer.transform(x)

    with pytest.raises(NotFittedError):
        transformer.inverse_transform(x)


def test_logscaling():
    tf = UtilityTransform(log=True)

    x = np.array([10, 100, -1])
    with pytest.raises(ValueError):
        tf.fit_transform(x)

    tf.fit_transform(x[:-1])


@pytest.fixture(params=[Ridge, LassoCV, Lasso, RidgeClassifierCV, XGBClassifier, LGBMClassifier])
def classifier(request):
    return request.param()


def test_simple(classifier, binary_Xy):
    model = PrePostProcessModel(classifier)
    model.fit(*binary_Xy)


def test_raise_not_fitting(classifier, binary_Xy):
    model = PrePostProcessModel(classifier)
    with pytest.raises(NotFittedError):
        model.predict(*binary_Xy)


def test_serializable(classifier, binary_Xy, tmpdir):
    model = PrePostProcessModel(classifier, input_scaling='standard')
    model.fit(*binary_Xy)
    y_pred = model.predict(binary_Xy[0])
    import joblib
    path = os.path.join(tmpdir, 'clf.joblib')
    joblib.dump(model, path)

    del model
    model_loaded = joblib.load(path)
    y_loaded = model_loaded.predict(binary_Xy[0])
    assert np.array_equal(y_pred, y_loaded)
