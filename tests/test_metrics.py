import numpy as np
from parameterized import parameterized
from sklearn.metrics import accuracy_score

from vivid.metrics import binary_metrics, regression_metrics


@parameterized.expand([
    (.05,),
    (.5,),
    (.95,)
])
def test_binary_metrics(t):
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([.9, .9, 1., .9, .1, .1])

    score = binary_metrics(y_true, y_pred, threshold=t)

    pred_label = np.where(y_pred > t, 1, 0)
    assert score.get('accuracy_score') == accuracy_score(y_true, pred_label)


def test_regression_metrics():
    y_true = np.random.uniform(size=(100,))
    y_pred = y_true + 1.
    regression_metrics(y_true, y_pred)
