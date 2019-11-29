from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseOutOfFoldFeature


class RFClassifierFeatureOutOfFold(BaseOutOfFoldFeature):
    model_class = RandomForestClassifier
    initial_params = {
        'criterion': 'gini',
        'class_weight': 'balanced'
    }


class RFRegressorFeatureOutOfFold(BaseOutOfFoldFeature):
    model_class = RandomForestRegressor
    initial_params = {
        'criterion': 'mse',
    }
