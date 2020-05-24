from sklearn.linear_model import LogisticRegression, Ridge

from .base import TunerBlock


class TunedLogisticBlock(TunerBlock):
    model_class = LogisticRegression
    initial_params = {
        'solver': 'liblinear',
        'penalty': 'l2',
        'input_scaling': 'standard',
    }

    def generate_model_class_try_params(self, trial):
        return {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        }


class TunedRidgeBlock(TunerBlock):
    model_class = Ridge
    initial_params = {
        'input_scaling': 'standard'
    }

    def generate_model_class_try_params(self, trial):
        return {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e2),
        }
