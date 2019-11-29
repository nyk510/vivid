from copy import deepcopy

import xgboost as xgb

from vivid.out_of_fold.base import BaseOptunaOutOfFoldFeature
from .mixins import FeatureImportanceMixin, BoostingOufOfFoldFeatureSet, get_boosting_parameter_suggestions


class XGBoostClassifierOutOfFold(BoostingOufOfFoldFeatureSet):
    eval_metric = 'logloss'
    model_class = xgb.XGBClassifier
    initial_params = {
        'learning_rate': .05,
        'reg_lambda': 1e-2,
        'n_estimators': 1000,
        'colsample_bytree': .8,
        'subsample': .7,
    }


class XGBoostRegressorOutOfFold(BoostingOufOfFoldFeatureSet):
    model_class = xgb.XGBRegressor
    initial_params = {
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'learning_rate': .2,
        'reg_lambda': 1e-2,
        'n_estimators': 1000,
        'colsample_bytree': .8,
        'subsample': .7,
    }


class OptunaXGBRegressionOutOfFold(FeatureImportanceMixin, BaseOptunaOutOfFoldFeature):
    model_class = xgb.XGBRegressor
    initial_params = deepcopy(XGBoostRegressorOutOfFold.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param


class OptunaXGBClassifierOutOfFold(FeatureImportanceMixin, BaseOptunaOutOfFoldFeature):
    model_class = xgb.XGBClassifier
    initial_params = deepcopy(XGBoostClassifierOutOfFold.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param
