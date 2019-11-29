import lightgbm as lgbm

from .mixins import BoostingOufOfFoldFeatureSet


class LGBMClassifierOutOfFold(BoostingOufOfFoldFeatureSet):
    model_class = lgbm.LGBMClassifier
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


class LGBMRegressorOutOfFold(BoostingOufOfFoldFeatureSet):
    model_class = lgbm.LGBMRegressor
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
