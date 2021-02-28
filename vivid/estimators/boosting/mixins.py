from copy import deepcopy
from typing import Union

from vivid.backends import ExperimentBackend
from .helpers import LogEvaluationCallback
from ..base import MetaBlock, TunerBlock


class BoostingEarlyStoppingMixin:
    """Override fit parameter for GBDT models (like XGBoost, LightGBM, ...)"""
    early_stopping_rounds = 200
    default_eval_metric = None
    fit_verbose = 200

    def get_fit_params_on_each_fold(self: Union['BoostingEarlyStoppingMixin', MetaBlock],
                                    model_params, training_set, validation_set, indexes_set,
                                    experiment: ExperimentBackend):
        params = super(BoostingEarlyStoppingMixin, self) \
            .get_fit_params_on_each_fold(model_params, training_set, validation_set, indexes_set, experiment)

        model_params = deepcopy(model_params)
        eval_metric = model_params.pop('eval_metric', self.default_eval_metric)
        add_params = dict(
            early_stopping_rounds=self.early_stopping_rounds,
            eval_set=[validation_set],
            eval_metric=eval_metric,
            verbose=100000,  # stop default_loader console log
            callbacks=[
                # write GBDT logging to the output log file
                LogEvaluationCallback(logger=experiment.logger,
                                      period=self.fit_verbose, )
            ]
        )
        params.update(add_params)
        return params


class BaseBoostingBlock(BoostingEarlyStoppingMixin, MetaBlock):
    pass


class TunedBoostingBlock(BoostingEarlyStoppingMixin, TunerBlock):
    pass
