# coding: utf-8
"""
"""

import os
from typing import Callable

from .backends import DataFrameBackend, ExperimentBackend
from .utils import import_string

__author__ = "nyk510"


class Settings:
    """store settings whole training"""

    RANDOM_SEED = int(os.getenv('VIVID_RANDOM_SEED', 19))
    N_FOLDS = int(os.getenv('VIVID_N_FOLDS', 5))

    LOG_LEVEL = os.getenv('VIVID_LOG_LEVEL', 'INFO')
    TXT_LOG_LEVEL = os.getenv('VIVID_TXT_LOG_LEVEL', 'DEBUG')

    CACHE_ON_TRAIN = os.getenv('VIVID_CACHE_ON_TRAIN', 'true') == 'true'
    CACHE_ON_TEST = os.getenv('VIVID_CACHE_ON_TEST', 'true') == 'true'
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.vivid')

    # using csv save / load backend class
    DATAFRAME_BACKEND = 'vivid.backends.dataframes.JoblibBackend'

    EXPERIMENT_BACKEND = 'vivid.backends.experiments.LocalExperimentBackend'
    COMET_API_KEY = os.getenv('VIVID_COMET_API_KEY', None)


class CacheLoader:
    def __init__(self, default_loader):
        self.default_loader = default_loader
        self.backend = None

    def __call__(self, name=None):
        backend = self.backend

        if name or not backend:
            backend = import_string(name or self.default_loader())()
            if not (self.backend or name):
                self.backend = backend
        return backend


get_dataframe_backend = CacheLoader(
    default_loader=lambda: Settings.DATAFRAME_BACKEND)  # type: Callable[[str], DataFrameBackend]
get_experiment_backend = CacheLoader(
    default_loader=lambda: Settings.EXPERIMENT_BACKEND)  # type: Callable[[str], ExperimentBackend]
