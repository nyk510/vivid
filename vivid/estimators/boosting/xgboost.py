from copy import deepcopy

import xgboost as xgb

from .helpers import get_boosting_parameter_suggestions
from .mixins import TunedBoostingBlock, BaseBoostingBlock


class XGBClassifierBlock(BaseBoostingBlock):
    default_eval_metric = 'logloss'
    model_class = xgb.XGBClassifier
    initial_params = {
        'learning_rate': .1,
        'reg_lambda': 1.,
        'n_estimators': 1000,
        'colsample_bytree': .5,
        'subsample': .7,
        'max_depth': 5,
        'min_child_weight': 1,
    }


class XGBRegressorBlock(BaseBoostingBlock):
    model_class = xgb.XGBRegressor
    default_eval_metric = 'rmse'
    initial_params = {
        'objective': 'reg:squarederror',
        'learning_rate': .1,
        'colsample_bytree': .5,
        'subsample': .7,
        'reg_lambda': 1.,
        'max_depth': 5,
        'min_child_weight': 1,
        'n_estimators': 10000,
    }


class TunedXGBRegressorBlock(TunedBoostingBlock):
    model_class = xgb.XGBRegressor
    default_eval_metric = 'rmse'
    initial_params = deepcopy(XGBRegressorBlock.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param


class TunedXGBClassifierBlock(TunedBoostingBlock):
    model_class = xgb.XGBClassifier
    default_eval_metric = 'logloss'
    initial_params = deepcopy(XGBClassifierBlock.initial_params)

    def generate_model_class_try_params(self, trial):
        param = get_boosting_parameter_suggestions(trial)
        param['n_jobs'] = 1
        return param
