from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import MetaBlock


class RFClassifierBlock(MetaBlock):
    model_class = RandomForestClassifier
    initial_params = {
        'criterion': 'gini',
        'class_weight': 'balanced'
    }


class RFRegressorBlock(MetaBlock):
    model_class = RandomForestRegressor
    initial_params = {
        'criterion': 'mse',
    }
