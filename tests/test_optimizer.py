import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from vivid.metrics import quadratic_weighted_kappa
from vivid.optimizers import BinsOptimizer


@pytest.fixture()
def state():
    return np.random.RandomState(seed=41)


class TestOptimizers:

    @pytest.mark.parametrize('obj', [
        quadratic_weighted_kappa, accuracy_score, f1_score
    ])
    def test_run(self, obj, state):
        optim = BinsOptimizer(obj, minimize=False)
        y_pred = state.uniform(size=1000)
        y_true = np.where(y_pred < .5, 0, 1)

        y_pred = y_pred * .5

        before = obj(y_true, np.round(y_pred))
        optim.fit(y_true, y_pred)
        b = optim.predict(y_pred)
        after = obj(y_true, b)

        # 前より良くなる
        assert after > before

        # なんならもともと 1. が出るはずなので .99 は超える
        assert after > .99
