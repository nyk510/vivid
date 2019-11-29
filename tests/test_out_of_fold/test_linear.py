from vivid.out_of_fold.linear import RidgeOutOfFold, LogisticOutOfFold
from .test_base import get_binary


class TestLinear:
    def setup_method(self):
        df, y = get_binary()
        self.df = df
        self.y = y

    def test_ridge(self):
        oof = RidgeOutOfFold(name='test_ridge', n_trials=10)
        oof.fit(self.df, self.y)

    def test_logistic(self):
        oof = LogisticOutOfFold(name='test_logistic', n_trials=10)
        oof.fit(self.df, self.y)
