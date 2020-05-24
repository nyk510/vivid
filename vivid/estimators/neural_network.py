from typing import Tuple

import numpy as np
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight

from vivid.estimators.base import MetaBlock
from vivid.sklearn_extend.neural_network import ScikitKerasClassifier, SKerasRegressor, ROCAucCallback


class BaseSkerasBlock(MetaBlock):
    initial_params = {
        'input_scaling': True,
        'epochs': 30,
        'batch_size': 128,
        'workers': -1
    }

    def get_keras_callbacks(self, training_set, validation_set):
        return [
            ReduceLROnPlateau(patience=5, verbose=1)
        ]

    def get_fit_params_on_each_fold(self,
                                    model_params: dict,
                                    training_set: Tuple[np.ndarray, np.ndarray],
                                    validation_set: Tuple[np.ndarray, np.ndarray],
                                    indexes_set: Tuple[np.ndarray, np.ndarray],
                                    experiment) -> dict:
        params = super(BaseSkerasBlock, self).get_fit_params_on_each_fold(
            model_params=model_params,
            training_set=training_set,
            validation_set=validation_set,
            indexes_set=indexes_set,
            experiment=experiment)

        add_params = {
            'callbacks': self.get_keras_callbacks(training_set, validation_set),
            'validation_data': validation_set,
        }

        params.update(add_params)
        return params


class KerasClassifierBlock(BaseSkerasBlock):
    model_class = ScikitKerasClassifier

    def get_keras_callbacks(self, training_set, validation_set):
        return [
            *super(KerasClassifierBlock, self).get_keras_callbacks(training_set, validation_set),
            ROCAucCallback(training_data=training_set, validation_data=validation_set),
        ]

    def get_fit_params_on_each_fold(self,
                                    model_params: dict,
                                    training_set: Tuple[np.ndarray, np.ndarray],
                                    validation_set: Tuple[np.ndarray, np.ndarray],
                                    indexes_set: Tuple[np.ndarray, np.ndarray],
                                    experiment) -> dict:
        params = super(KerasClassifierBlock, self) \
            .get_fit_params_on_each_fold(model_params, training_set, validation_set, indexes_set, experiment)

        y = training_set[1]
        weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
        params['class_weight'] = weight
        return params


class KerasRegressorBlock(BaseSkerasBlock):
    model_class = SKerasRegressor
