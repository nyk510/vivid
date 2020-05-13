import lightgbm as lgbm

from .mixins import BoostingOutOfFoldFeature


class LGBMClassifierOutOfFold(BoostingOutOfFoldFeature):
    model_class = lgbm.LGBMClassifier
    default_eval_metric = 'logloss'
    initial_params = {
        'learning_rate': .1,
        'reg_lambda': 1e-2,
        'n_estimators': 1000,
        'min_child_samples': 5,
        'colsample_bytree': .8,
        'subsample': .7,
        'metric': 'logloss',
        'num_leaves': 31,
    }


class LGBMRegressorOutOfFold(BoostingOutOfFoldFeature):
    model_class = lgbm.LGBMRegressor
    default_eval_metric = 'rmse'
    initial_params = {
        'learning_rate': .1,
        'reg_lambda': 1e-2,
        'n_estimators': 300,
        'min_child_samples': 5,
        'colsample_bytree': .8,
        'subsample': .7,
        'metric': 'rmse',
        'num_leaves': 31,
    }
