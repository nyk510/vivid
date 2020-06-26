from copy import deepcopy

import lightgbm as lgbm

from .mixins import BaseBoostingBlock


def _to_gpu(params: dict) -> dict:
    p = deepcopy(params)
    p.update({
        'tree_method': 'gpu_hist',
        'device': 'gpu'
    })
    return p


class ParameterStore:
    class Classification:
        cpu = {
            'learning_rate': .1,
            'reg_lambda': 1e-2,
            'n_estimators': 1000,
            'min_child_samples': 5,
            'colsample_bytree': .8,
            'subsample': .7,
            'num_leaves': 31,
            'importance_type': 'gain'
        }
        gpu = _to_gpu(cpu)

    class Regression:
        cpu = {
            'learning_rate': .1,
            'reg_lambda': 1e-2,
            'n_estimators': 300,
            'min_child_samples': 5,
            'colsample_bytree': .8,
            'subsample': .7,
            'num_leaves': 31,
            'importance_type': 'gain'
        }

        gpu = _to_gpu(cpu)


class LGBMClassifierBlock(BaseBoostingBlock):
    model_class = lgbm.LGBMClassifier
    default_eval_metric = 'logloss'
    initial_params = deepcopy(ParameterStore.Classification.cpu)


class LGBMRegressorBlock(BaseBoostingBlock):
    model_class = lgbm.LGBMRegressor
    default_eval_metric = 'rmse'
    initial_params = deepcopy(ParameterStore.Regression.cpu)
