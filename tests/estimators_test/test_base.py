import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import KFold, StratifiedKFold

from vivid.estimators import boosting
from vivid.estimators.ensumble import RFRegressorBlock
from vivid.estimators.kneighbor import TunedKNNRegressorBlock


def test_serializing(regression_data):
    feat_none_save = TunedKNNRegressorBlock(name='serialize_0', n_trials=1)
    feat_none_save.fit(*regression_data)

    with pytest.raises(NotFittedError):
        feat_none_save.load_best_models()


# def test_not_recoding(regression_data):
#     df, y = regression_data
#     feat_not_recoding_root = TunedKNNRegressorBlock(parent=base_feat, name='not_save_parent', n_trials=1)
#     feat_not_recoding_root.fit(df, y)
#     with pytest.raises(NotFittedError):
#         feat_not_recoding_root.load_best_models()
#

def test_use_cache(regression_data):
    df, y = regression_data
    feat = TunedKNNRegressorBlock(parent=None, name='knn', n_trials=1)

    output_df = feat.fit(df, y)
    assert feat.feat_on_train_df is not None
    output_repeat_df = feat.fit(df, y)

    assert output_df is output_repeat_df, feat


# @pytest.mark.parametrize('model_class', [boosting.XGBRegressorBlock, boosting.XGBClassifierBlock])
# def test_recording(model_class: Type[MetaBlock], regression_data, binary_data):
#     recording_feature = RecordingFeature()
#     clf = model_class(parent=recording_feature, name='serialize_1')
#
#     with pytest.raises(NotFittedError):
#         clf.load_best_models()
#
#     # 学習前なのでモデルの重みはあってはならない
#     assert not os.path.exists(clf.model_param_path), clf
#
#     if clf.is_regression_model:
#         df, y = regression_data
#     else:
#         df, y = binary_data
#     clf.fit(df, y)
#
#     assert clf.is_recording, clf
#     assert clf.is_train_finished, clf
#
#     # 学習後なのでモデルの重みが無いとだめ
#     assert os.path.exists(clf.model_param_path), clf
#
#     # モデル読み込みと予測が可能
#     clf.load_best_models()
#     pred1 = clf.predict(df)
#
#     # 再度定義しなおしても推論は可能 (ローカルからモデルを呼び出して来ることができる)
#     clf = model_class(parent=recording_feature, name='serialize_1')
#     pred2 = clf.predict(df)
#     assert pred1.equals(pred2), (pred1, pred2)


def test_add_sample_weight(regression_data):
    df, y = regression_data
    sample_weight = df.values[:, 0]
    model = RFRegressorBlock(name='rf', sample_weight=sample_weight)
    model.fit(df, y)

    for clf, (idx_train, idx_valid) in zip(model._fitted_models, model.get_fold_splitting(df.values, y)):
        assert np.array_equal(clf.fit_params_.get('sample_weight', None), sample_weight[idx_train])


@pytest.mark.parametrize('cv', [
    KFold(n_splits=2),
    StratifiedKFold(n_splits=10, shuffle=True, random_state=71),  # Must set random state
])
def test_custom_cv_class(cv, binary_data):
    df, y = binary_data
    clf = boosting.XGBClassifierBlock(name='xgb', cv=cv)

    for origin, model in zip(cv.split(df.values, y), clf.get_fold_splitting(df.values, y)):
        assert np.array_equal(origin[0], model[0])
        assert np.array_equal(origin[1], model[1])


def test_custom_cv_as_list():
    """can set custom cv as list of _fit_core / test indexes"""
    cv = [
        [1, 2, 3], [4, 5],
        [2, 4, 5], [1, 3]
    ]
    clf = boosting.XGBClassifierBlock(name='xgb', cv=cv)

    X = np.random.uniform(size=(5, 10))
    y = np.random.uniform(size=(5,))

    for cv_idxes, clf_indexes in zip(cv, clf.get_fold_splitting(X, y)):
        assert np.array_equal(cv_idxes[0], clf_indexes[0])
        assert np.array_equal(cv_idxes[1], clf_indexes[1])


@pytest.mark.parametrize('n_folds, expected', [
    (None, 5),
    (-1, 0),
    (0, 0),
    (.5, 1),
    (5, 5),
    (6, 5)
])
def test_short_n_fold(regression_data, n_folds, expected):
    input_df, y = regression_data
    model = boosting.XGBRegressorBlock(name='xgb', cv=5)
    fitted_models, _ = model.run_oof_train(input_df.values, y, {}, n_max=n_folds)

    assert len(fitted_models) == expected


def test_updated_model_parameters_add_init_params(regression_data):
    add_params = {
        'n_estimators': 1,
        'colsample_bytree': .9
    }
    model = boosting.XGBRegressorBlock(name='xgb', add_init_param=add_params)
    input_df, y = regression_data
    model.fit(input_df, y)

    for m in model._fitted_models:
        for key, value in add_params.items():
            assert getattr(m.fitted_model_, key) == value


@pytest.mark.parametrize('metric_func', [
    mean_absolute_error,
    mean_squared_log_error
])
def test_optuna_change_metric(metric_func, regression_data):
    df, y = regression_data
    scoring = make_scorer(metric_func, greater_is_better=False)
    model = TunedKNNRegressorBlock(name='optuna', n_trials=1, scoring=scoring,
                                   scoring_strategy='fold')
    model.fit(df, y)

    X = df.values
    scores = []
    for clf, (idx_train, idx_valid) in zip(model._fitted_models, model.get_fold_splitting(X, y)):
        pred_i = clf.predict(X[idx_valid])
        score = metric_func(y[idx_valid], pred_i)
        scores.append(score)
    score = np.mean(scores)
    np.testing.assert_almost_equal(-score, model.study.best_value, decimal=7)


def test_find_best_value(regression_data):
    model = TunedKNNRegressorBlock(name='optuna_test', n_trials=10, scoring='neg_mean_absolute_error')
    df, y = regression_data
    model.fit(df, y)

    trial_df = model.study.trials_dataframe()
    assert trial_df['value'].max() == model.study.best_value


def test_change_scoring_strategy(regression_data):
    """check same scoring value in convex objective"""
    df, y = regression_data
    model = TunedKNNRegressorBlock(name='test', n_trials=1,
                                   scoring='neg_root_mean_squared_error',
                                   scoring_strategy='whole')
    oof_df = model.fit(df, y)
    from sklearn.metrics import mean_squared_error

    assert - mean_squared_error(y, oof_df.values[:, 0]) ** .5 == model.study.best_value


def test_custom_scoring_metric(regression_data):
    df, y = regression_data
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    model = TunedKNNRegressorBlock(name='optuna', n_trials=10, scoring=scoring)
    model.fit(df, y)
    log_df = model.study.trials_dataframe()
    assert (log_df['value'] > 0).sum() == 0, log_df
