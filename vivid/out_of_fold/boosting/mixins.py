from copy import deepcopy
from typing import Union

from .helpers import logging_evaluation
from ..base import BaseOutOfFoldFeature, GenericOutOfFoldFeature, GenericOutOfFoldOptunaFeature


class BoostingEarlyStoppingMixin:
    """Override fit parameter for GBDT models (like XGBoost, LightGBM, ...)"""
    early_stopping_rounds = 100
    default_eval_metric = None
    fit_verbose = 100

    def get_fit_params_on_each_fold(self: Union['BoostingEarlyStoppingMixin', BaseOutOfFoldFeature],
                                    model_params, training_set, validation_set, indexes_set):
        params = super(BoostingEarlyStoppingMixin, self) \
            .get_fit_params_on_each_fold(model_params, training_set, validation_set, indexes_set)
        model_params = deepcopy(model_params)
        eval_metric = model_params.pop('eval_metric', self.default_eval_metric)
        add_params = dict(
            early_stopping_rounds=self.early_stopping_rounds,
            eval_set=[validation_set],
            eval_metric=eval_metric,
            verbose=0,  # stop default_loader console log
            callbacks=[
                # write GBDT logging to the output log file
                logging_evaluation(logger=self.logger, period=self.fit_verbose)
            ]
        )
        params.update(add_params)
        return params


class BoostingOutOfFoldFeature(BoostingEarlyStoppingMixin, GenericOutOfFoldFeature):
    pass


class BoostingOptunaFeature(BoostingEarlyStoppingMixin, GenericOutOfFoldOptunaFeature):
    pass
