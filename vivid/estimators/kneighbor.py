# coding: utf-8
"""
"""

from optuna.trial import Trial
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import MetaBlock, TunerBlock


class KNNClassifierBlock(MetaBlock):
    model_class = KNeighborsClassifier
    initial_params = {
        'n_neighbors': 5
    }


class KNNRegressorBlock(MetaBlock):
    model_class = KNeighborsRegressor
    initial_params = {
        'n_neighbors': 5
    }


class TunedKNNRegressorBlock(TunerBlock):
    model_class = KNeighborsRegressor

    def generate_model_class_try_params(self, trial: Trial):
        params = {
            'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
            'p': trial.suggest_uniform('p', 1, 4),
            'n_neighbors': int(trial.suggest_int('n_neighbors', 5, 30)),
            'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree'])
        }

        if 'tree' in params.get('algorithm', None):
            params['leaf_size'] = int(trial.suggest_int('leaf_size', 10, 200))

        return params
