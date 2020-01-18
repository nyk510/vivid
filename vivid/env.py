# coding: utf-8
"""
"""

import os

__author__ = "nyk510"


class Settings:
    """
    学習に関連するパラメータを管理するためのクラス
    """

    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 19))
    N_FOLDS = int(os.getenv('N_FOLDS', 5))

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    TXT_LOG_LEVEL = os.getenv('TXT_LOG_LEVEL', 'DEBUG')

    CACHE_ON_TRAIN = os.getenv('VIVID_CACHE_ON_TRAIN', 'true') == 'true'
    CACHE_ON_TEST = os.getenv('VIVID_CACHE_ON_TEST', 'true') == 'true'
