"""
sklearn extend のテストコード
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeClassifierCV
from sklearn.utils.validation import NotFittedError

from vivid.sklearn_extend import UtilityTransform, PrePostProcessModel
from .utils import is_close_to_zero


class TestTransform(object):
    @pytest.mark.parametrize('log,scaling', [
        (True, None),
        (False, None),
        (True, 'standard')
    ])
    def test_basic(self, log, scaling):
        transformer = UtilityTransform()

        x = np.random.uniform(size=100)
        x_edited = transformer.fit_transform(x)
        assert is_close_to_zero(x, x_edited)

        transformer = UtilityTransform(log=log, scaling=scaling)
        x_trans = transformer.fit_transform(x)
        x_inv = transformer.inverse_transform(x_trans)
        assert is_close_to_zero(x, x_inv)

    def test_out_of_scaling_string(self):
        transformer = UtilityTransform(log=True, scaling='hogehoge')
        assert transformer.scaling is None

    def test_not_fitteed_error(self):
        transformer = UtilityTransform()
        x = np.random.uniform(size=10)
        with pytest.raises(NotFittedError):
            transformer.transform(x)

        with pytest.raises(NotFittedError):
            transformer.inverse_transform(x)

    def test_logscaling(self):
        tf = UtilityTransform(log=True)

        x = np.array([10, 100, -1])
        with pytest.raises(ValueError):
            tf.fit_transform(x)

        tf.fit_transform(x[:-1])


class TestRecordingModel(object):
    @pytest.mark.parametrize('model_class', [
        Ridge, LassoCV, Lasso, RidgeClassifierCV
    ])
    def test_simple(self, model_class):
        model = PrePostProcessModel(model_class=model_class, model_params=None)
        assert not model.is_recording

        x, y = np.random.uniform(size=(10, 10)), np.random.random_integers(0, 1, size=(10,))
        model.fit(x, y)

    def test_raise_not_fitting(self):
        model = PrePostProcessModel(model_class=Lasso)
        x, y = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        with pytest.raises(NotFittedError):
            model.predict(x)

    def test_save_and_read(self, output_dir):
        model_class = Lasso
        model = PrePostProcessModel(model_class=model_class, model_params=None, output_dir=output_dir,
                                    input_scaling='standard')

        assert model.is_recording

        x, y = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))

        with pytest.raises(FileNotFoundError):
            model.load_trained_model()
        model.fit(x, y)

        pred_1 = model.predict(x)

        model.load_trained_model()
        pred_2 = model.predict(x)

        assert is_close_to_zero(pred_1, pred_2)

        pred_3 = model.predict(x, prob=True)
        assert not is_close_to_zero(x, pred_3)
