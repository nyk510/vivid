import numpy as np


def is_close_to_zero(x, y):
    return np.sum(abs(x - y) > 1e-8) == 0
