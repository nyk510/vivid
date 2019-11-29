import pandas as pd
from sklearn.datasets import load_boston

from vivid.out_of_fold import boosting
from vivid.out_of_fold.kneighbor import OptunaKNeighborRegressorOutOfFold
from .fixtures import SampleFeature, df_train, y

base_feat = SampleFeature()


def get_boston():
    X, y = load_boston(True)
    return pd.DataFrame(X), y


class TestKneighbors(object):
    def test_optuna(self):
        df_train, y = get_boston()
        feat = OptunaKNeighborRegressorOutOfFold(parent=base_feat, name='test_kn_1', n_trials=10)
        feat.fit(df_train, y, force=True)
        assert feat.study is not None
        assert isinstance(feat.study.trials_dataframe(), pd.DataFrame)


class TestXGBoost(object):
    def test_xgb_feature(self):
        feat = boosting.XGBoostRegressorOutOfFold(parent=base_feat, name='test_xgb')
        feat.fit(df_train, y, force=True)

    def test_optuna_xgb(self):
        feat = boosting.OptunaXGBRegressionOutOfFold(parent=None, name='test_xgb_optuna', n_trials=10)
        df_train, y = get_boston()
        feat.fit(df_train, y, force=True)

        assert feat.study.best_params is not None
        assert feat.study.best_value is not None
        assert feat.study is not None
        assert isinstance(feat.study.trials_dataframe(), pd.DataFrame)
        pred = feat.predict(df_train)
        assert pred.values.shape[0] == y.shape[0]


class TestLightGBM(object):
    def test_lgbm_feature(self):
        feat = boosting.LGBMRegressorOutOfFold(parent=base_feat, name='test_lightgbm')
        feat.fit(df_train, y, force=True)
