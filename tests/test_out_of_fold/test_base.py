import os
import shutil
from typing import Type
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import KFold, StratifiedKFold

from tests.conftest import SampleFeature, RECORDING_DIR, RecordingFeature
from vivid.out_of_fold import boosting
from vivid.out_of_fold.base import NotFittedError, BaseOutOfFoldFeature
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold
from vivid.out_of_fold.kneighbor import OptunaKNeighborRegressorOutOfFold
from vivid.out_of_fold.linear import RidgeOutOfFold

base_feat = SampleFeature()


def get_boston():
    X, y = load_boston(True)
    return pd.DataFrame(X), y


def get_binary():
    x, y = load_breast_cancer(True)
    return pd.DataFrame(x), y


class TestCore(TestCase):
    def setUp(self):
        if os.path.exists(RECORDING_DIR):
            shutil.rmtree(RECORDING_DIR)

    def test_serializing(self):
        feat_none_save = OptunaKNeighborRegressorOutOfFold(name='serialize_0', n_trials=1)
        feat_none_save.fit(*get_boston())

        with pytest.raises(NotFittedError):
            feat_none_save.load_best_models()

    def test_not_recoding(self):
        feat_not_recoding_root = OptunaKNeighborRegressorOutOfFold(parent=base_feat, name='not_save_parent', n_trials=1)
        feat_not_recoding_root.fit(*get_boston())
        with pytest.raises(NotFittedError):
            feat_not_recoding_root.load_best_models()

    def test_use_cache(self):
        df, y = get_boston()
        feat = OptunaKNeighborRegressorOutOfFold(parent=None, name='knn', n_trials=1)

        output_df = feat.fit(df, y)
        self.assertIsNotNone(feat.feat_on_train_df, feat)
        output_repeat_df = feat.fit(df, y)
        self.assertIs(output_df, output_repeat_df, feat)

    @parameterized.expand([
        (RidgeOutOfFold, {'n_trials': 1}),
        (OptunaKNeighborRegressorOutOfFold, {'n_trials': 1}),
        (boosting.XGBoostRegressorOutOfFold,),
        (boosting.XGBoostClassifierOutOfFold,),
        (boosting.LGBMRegressorOutOfFold,),
        (boosting.LGBMClassifierOutOfFold,),
        (RFRegressorFeatureOutOfFold,),
    ])
    def test_recording(self, model_class: Type[BaseOutOfFoldFeature], params=None):
        if params is None: params = {}
        recording_feature = RecordingFeature()
        clf = model_class(parent=recording_feature, name='serialize_1', **params)

        with pytest.raises(NotFittedError):
            clf.load_best_models()

        # 学習前なのでモデルの重みはあってはならない
        assert not os.path.exists(clf.serializer_path), clf

        if clf.is_regression_model:
            df, y = get_boston()
        else:
            df, y = get_binary()
        clf.fit(df, y)

        assert clf.is_recording, clf
        assert clf.finish_fit, clf

        # 学習後なのでモデルの重みが無いとだめ
        assert os.path.exists(clf.serializer_path), clf

        # モデル読み込みと予測が可能
        clf.load_best_models()
        pred1 = clf.predict(df)

        # 再度定義しなおしても推論は可能 (ローカルからモデルを呼び出して来ることができる)
        clf = model_class(parent=recording_feature, name='serialize_1', **params)
        pred2 = clf.predict(df)
        assert pred1.equals(pred2), (pred1, pred2)

    def test_add_sample_weight(self):
        df, y = get_boston()
        sample_weight = df.values[:, 0]
        model = RFRegressorFeatureOutOfFold(name='rf', sample_weight=sample_weight)
        model.fit(df, y)

        for clf, (idx_train, idx_valid) in zip(model.fitted_models, model.get_fold_splitting(df.values, y)):
            assert np.array_equal(clf.fit_params_.get('sample_weight', None), sample_weight[idx_train])


@parameterized.expand([
    (KFold(n_splits=2),),
    (StratifiedKFold(n_splits=10, shuffle=True, random_state=71),),  # Must set random state
])
def test_custom_cv_class(cv):
    df, y = get_binary()
    clf = boosting.XGBoostClassifierOutOfFold(name='xgb', cv=cv)

    for origin, model in zip(cv.split(df.values, y), clf.get_fold_splitting(df.values, y)):
        assert np.array_equal(origin[0], model[0])
        assert np.array_equal(origin[1], model[1])


def test_custom_cv_as_list():
    """can set custom cv as list of train / test indexes"""
    cv = [
        [1, 2, 3], [4, 5],
        [2, 4, 5], [1, 3]
    ]
    clf = boosting.XGBoostClassifierOutOfFold(name='xgb', cv=cv)

    X = np.random.uniform(size=(5, 10))
    y = np.random.uniform(size=(5,))

    for cv_idxes, clf_indexes in zip(cv, clf.get_fold_splitting(X, y)):
        assert np.array_equal(cv_idxes[0], clf_indexes[0])
        assert np.array_equal(cv_idxes[1], clf_indexes[1])


def test_updated_model_parameters_add_init_params():
    add_params = {
        'n_estimators': 1,
        'colsample_bytree': .9
    }
    model = boosting.XGBoostRegressorOutOfFold(name='xgb', add_init_param=add_params)
    input_df, y = get_boston()
    model.fit(input_df, y)

    for m in model.fitted_models:
        for key, value in add_params.items():
            assert getattr(m.fitted_model_, key) == value


@pytest.mark.parametrize('metric_func', [
    mean_absolute_error,
    mean_squared_log_error
])
def test_optuna_change_metric(metric_func):
    df, y = get_boston()
    scoring = make_scorer(metric_func, greater_is_better=False)
    model = OptunaKNeighborRegressorOutOfFold(name='optuna', n_trials=1, scoring=scoring,
                                              scoring_strategy='fold')
    model.fit(df, y)

    X = df.values
    scores = []
    for clf, (idx_train, idx_valid) in zip(model.fitted_models, model.get_fold_splitting(X, y)):
        pred_i = clf.predict(X[idx_valid])
        score = metric_func(y[idx_valid], pred_i)
        scores.append(score)
    score = np.mean(scores)
    np.testing.assert_almost_equal(-score, model.study.best_value, decimal=7)


def test_find_best_value():
    model = OptunaKNeighborRegressorOutOfFold(name='optuna_test', n_trials=10, scoring='neg_mean_absolute_error')
    df, y = get_boston()
    model.fit(df, y)

    trial_df = model.study.trials_dataframe()
    assert trial_df['value'].max() == model.study.best_value


def test_change_scoring_strategy():
    """check same scoring value in convex objective"""
    df, y = get_boston()
    model = OptunaKNeighborRegressorOutOfFold(name='test', n_trials=1,
                                              scoring='neg_root_mean_squared_error',
                                              scoring_strategy='whole')
    oof_df = model.fit(df, y)
    from sklearn.metrics import mean_squared_error

    assert - mean_squared_error(y, oof_df.values[:, 0]) ** .5 == model.study.best_value


def test_custom_scoring_metric():
    df, y = get_boston()
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    model = OptunaKNeighborRegressorOutOfFold(name='optuna', n_trials=10, scoring=scoring)
    model.fit(df, y)
    log_df = model.study.trials_dataframe()
    assert (log_df['value'] > 0).sum() == 0, log_df


class TestKneighbors(object):
    def test_optuna(self):
        df_train, y = get_boston()
        feat = OptunaKNeighborRegressorOutOfFold(parent=None, name='test_kn_1', n_trials=1)
        feat.fit(df_train, y, force=True)

        feat = OptunaKNeighborRegressorOutOfFold(parent=base_feat, name='test_kn_1', n_trials=1)
        feat.fit(df_train, y, force=True)
        assert feat.study is not None
        assert isinstance(feat.study.trials_dataframe(), pd.DataFrame)


class TestXGBoost(object):
    def test_xgb_feature(self, train_data):
        train_df, y = train_data
        feat = boosting.XGBoostRegressorOutOfFold(parent=None, name='test_xgb')
        feat.fit(train_df, y, force=True)

    def test_optuna_xgb(self, train_data):
        feat = boosting.OptunaXGBRegressionOutOfFold(parent=None, name='test_xgb_optuna', n_trials=1)
        df_train, y = get_boston()
        feat.fit(df_train, y, force=True)

        assert feat.study.best_params is not None
        assert feat.study.best_value is not None
        assert feat.study is not None
        assert isinstance(feat.study.trials_dataframe(), pd.DataFrame)
        pred = feat.predict(df_train)
        assert pred.values.shape[0] == y.shape[0]


class TestLightGBM(object):
    def test_lgbm_feature(self, train_data):
        train_df, y = train_data
        feat = boosting.LGBMRegressorOutOfFold(parent=None, name='test_lightgbm')
        feat.fit(train_df, y, force=True)
