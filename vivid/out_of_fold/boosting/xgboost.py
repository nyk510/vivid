from copy import deepcopy

import xgboost as xgb

from .helpers import get_boosting_parameter_suggestions
from .mixins import BoostingOptunaFeature, BoostingOutOfFoldFeature


class XGBoostClassifierOutOfFold(BoostingOutOfFoldFeature):
    default_eval_metric = 'logloss'
    model_class = xgb.XGBClassifier
    initial_params = {
        'learning_rate': .1,
        'reg_lambda': 1e-2,
        'n_estimators': 1000,
        'colsample_bytree': .8,
        'subsample': .7,
    }


class XGBoostRegressorOutOfFold(BoostingOutOfFoldFeature):
    model_class = xgb.XGBRegressor
    default_eval_metric = 'rmse'
    initial_params = {
        'objective': 'reg:squarederror',
        'learning_rate': .1,
        'colsample_bytree': .7,
        'subsample': .8,
        'reg_lambda': 1.,
        'max_depth': 5,
        'min_child_weight': 1,
        'n_estimators': 10000,
    }


class OptunaXGBRegressionOutOfFold(BoostingOptunaFeature):
    model_class = xgb.XGBRegressor
    default_eval_metric = 'rmse'
    initial_params = deepcopy(XGBoostRegressorOutOfFold.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param


class OptunaXGBClassifierOutOfFold(BoostingOptunaFeature):
    model_class = xgb.XGBClassifier
    default_eval_metric = 'logloss'
    initial_params = deepcopy(XGBoostClassifierOutOfFold.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param
