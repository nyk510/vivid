import os
import shutil

import pandas as pd
import pytest
from sklearn.datasets import load_boston, load_breast_cancer

from tests.fixtures import SampleFeature, df_train, y, RecordingFeature, RECORDING_DIR
from vivid.out_of_fold import boosting
from vivid.out_of_fold.base import NotFittedError
from vivid.out_of_fold.kneighbor import OptunaKNeighborRegressorOutOfFold

base_feat = SampleFeature()


def get_boston():
    X, y = load_boston(True)
    return pd.DataFrame(X), y


def get_binary():
    x, y = load_breast_cancer(True)
    return pd.DataFrame(x), y


class TestCore(object):
    def setup_method(self):
        if os.path.exists(RECORDING_DIR):
            shutil.rmtree(RECORDING_DIR)

    def test_serialize(self):
        feat_none_save = OptunaKNeighborRegressorOutOfFold(name='serialize_0', n_trials=1)
        feat_none_save.fit(*get_boston())

        with pytest.raises(NotFittedError):
            feat_none_save.load_best_models()

        feat_not_recoding_root = OptunaKNeighborRegressorOutOfFold(parent=base_feat, name='not_save_parent', n_trials=1)
        feat_not_recoding_root.fit(*get_boston())
        with pytest.raises(NotFittedError):
            feat_not_recoding_root.load_best_models()

        recoding_feate = RecordingFeature()
        feat = OptunaKNeighborRegressorOutOfFold(parent=recoding_feate, name='serialize_1', n_trials=1)

        with pytest.raises(NotFittedError):
            feat.load_best_models()

        # 学習前なのでモデルの重みはあってはならない
        assert not os.path.exists(feat.serializer_path), feat
        feat.fit(*get_boston())

        assert feat.is_recording and feat.finish_fit, feat
        # 学習後なのでモデルの重みが無いとだめ
        assert os.path.exists(feat.serializer_path), feat

        # モデル読み込みと予測が可能
        feat.load_best_models()
        feat.predict(get_boston()[0])


class TestKneighbors(object):
    def test_optuna(self):
        df_train, y = get_boston()
        feat = OptunaKNeighborRegressorOutOfFold(parent=None, name='test_kn_1', n_trials=10)
        feat.fit(df_train, y, force=True)

        feat = OptunaKNeighborRegressorOutOfFold(parent=base_feat, name='test_kn_1', n_trials=10)
        feat.fit(df_train, y, force=True)
        assert feat.study is not None
        assert isinstance(feat.study.trials_dataframe(), pd.DataFrame)


class TestXGBoost(object):
    def test_xgb_feature(self):
        feat = boosting.XGBoostRegressorOutOfFold(parent=None, name='test_xgb')
        feat.fit(df_train, y, force=True)

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
