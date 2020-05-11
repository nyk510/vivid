from sklearn.linear_model import LogisticRegression, Ridge

from .base import GenericOutOfFoldOptunaFeature


class LogisticOutOfFold(GenericOutOfFoldOptunaFeature):
    initial_params = {
        'solver': 'liblinear',
        'penalty': 'l2',
        'input_scaling': 'standard',
        'n_jobs': 1,
    }
    model_class = LogisticRegression

    def generate_model_class_try_params(self, trial):
        return {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        }


class RidgeOutOfFold(GenericOutOfFoldOptunaFeature):
    model_class = Ridge

    def generate_model_class_try_params(self, trial):
        return {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e2),
        }
