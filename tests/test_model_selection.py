import numpy as np

from vivid.model_selection import ContinuousStratifiedFold


def test_continuous_stratified():
    y = [1, 1, 2, 2, 3, 3]
    y = np.asarray(y)
    fold = ContinuousStratifiedFold(n_splits=2, shuffle=False, q=10)
    cv = list(fold.split(y, y))

    for idx_tr, idx_val in cv:
        for i in [1, 2, 3]:
            y_tr = y[idx_tr]
            assert (y_tr == i).sum() == 1
