from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import GenericOutOfFoldFeature


class RFClassifierFeatureOutOfFold(GenericOutOfFoldFeature):
    model_class = RandomForestClassifier
    initial_params = {
        'criterion': 'gini',
        'class_weight': 'balanced'
    }


class RFRegressorFeatureOutOfFold(GenericOutOfFoldFeature):
    model_class = RandomForestRegressor
    initial_params = {
        'criterion': 'mse',
    }
