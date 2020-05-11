# coding: utf-8
"""
"""

import os

from .backends import AbstractBackend
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
    DATAFRAME_BACKEND = 'vivid.backends.JoblibBackend'


def get_dataframe_backend(name=None) -> AbstractBackend:
    loaded_backend = get_dataframe_backend.backend

    if name or not loaded_backend:
        loaded_backend = import_string(name or Settings.DATAFRAME_BACKEND)()
        if not (get_dataframe_backend.backend or name):
            get_dataframe_backend.backend = loaded_backend
    return loaded_backend


get_dataframe_backend.backend = None
