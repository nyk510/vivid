import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import KFold, StratifiedKFold

from vivid.estimators import boosting
from vivid.estimators.base import MetaBlock
from vivid.estimators.kneighbor import TunedKNNRegressorBlock


def test_add_sample_weight(regression_data, experiment):
    """sample weight がちゃんと model fit に伝搬しているかの確認"""
    class RecordRidge(Ridge):
        def fit(self, *args, **kwargs):
            self.args = args
            self.kwrgs = kwargs
            return super(RecordRidge, self).fit(*args, **kwargs)

    class TestBlock(MetaBlock):
        model_class = RecordRidge

    df, y = regression_data
    sample_weight = df.values[:, 0]
    model = TestBlock(name='test',
                      sample_weight=sample_weight)
    model.fit(df, y, experiment)

    for clf, (idx_train, idx_valid) in zip(model._fitted_models, model.get_fold_splitting(df.values, y)):
        params = clf.fitted_model_.kwrgs
        assert np.array_equal(params.get('sample_weight', None), sample_weight[idx_train])


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
    """can set custom cv as list of fit / test indexes"""
    cv = [
        [[1, 2, 3], [4, 5]],
        [[2, 4, 5], [1, 3]]
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


def test_updated_model_parameters_add_init_params(regression_data, experiment):
    add_params = {
        'n_estimators': 1,
        'colsample_bytree': .9
    }
    model = boosting.XGBRegressorBlock(name='xgb', add_init_param=add_params)
    input_df, y = regression_data
    model.fit(input_df, y, experiment)

    for m in model._fitted_models:
        for key, value in add_params.items():
            assert getattr(m.fitted_model_, key) == value


@pytest.mark.parametrize('metric_func', [
    mean_absolute_error,
    mean_squared_log_error
])
def test_optuna_change_metric(metric_func, regression_data, experiment):
    df, y = regression_data
    scoring = make_scorer(metric_func, greater_is_better=False)
    model = TunedKNNRegressorBlock(name='optuna', n_trials=1, scoring=scoring,
                                   scoring_strategy='fold')
    model.fit(df, y, experiment)

    X = df.values
    scores = []
    for clf, (idx_train, idx_valid) in zip(model._fitted_models, model.get_fold_splitting(X, y)):
        pred_i = clf.predict(X[idx_valid])
        score = metric_func(y[idx_valid], pred_i)
        scores.append(score)
    score = np.mean(scores)
    np.testing.assert_almost_equal(-score, model.study.best_value, decimal=7)


def test_find_best_value(regression_data, experiment):
    model = TunedKNNRegressorBlock(name='optuna_test', n_trials=10, scoring='neg_mean_absolute_error')
    df, y = regression_data
    model.fit(df, y, experiment)

    trial_df = model.study.trials_dataframe()
    assert trial_df['value'].max() == model.study.best_value


def test_change_scoring_strategy(regression_data, experiment):
    """check same scoring value in convex objective"""
    df, y = regression_data
    model = TunedKNNRegressorBlock(name='test', n_trials=1,
                                   scoring='neg_root_mean_squared_error',
                                   scoring_strategy='whole')
    oof_df = model.fit(df, y, experiment)
    from sklearn.metrics import mean_squared_error

    assert - mean_squared_error(y, oof_df.values[:, 0]) ** .5 == model.study.best_value


def test_custom_scoring_metric(regression_data, experiment):
    df, y = regression_data
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    model = TunedKNNRegressorBlock(name='optuna', n_trials=10, scoring=scoring)
    model.fit(df, y, experiment)
    log_df = model.study.trials_dataframe()
    assert (log_df['value'] > 0).sum() == 0, log_df
