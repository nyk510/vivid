# coding: utf-8
"""
"""

import os
from contextlib import contextmanager
from typing import Union, List

import numpy as np
import pandas as pd

from .backends.experiments import LocalExperimentBackend
from .env import Settings, get_dataframe_backend
from .utils import get_logger, timer


class AbstractFeature(object):
    """Abstract class for all feature.

    The feature quantity consists of two main methods.

    * fit: the learning phase, which the internal state is changed based on the input features
    * predict: the prediction phase, which creates a new feature based on the learned state.

    In order to be consistent with the fit/predict conversions, both methods eventually call the call method.
    If you want to create a new feature, override it.

    It is not recommended to override the fit/predict method. This is to be consistent with the conversion
    in fit/predict method. We believe that the difference between the code at the time of prediction execution and
    the feature creation code in the learning phase is **the biggest cause of inconsistency** between the training and
    prediction feature.
    """

    # if set True, allow save to local.
    allow_save_local = True

    def __init__(self,
                 name: str,
                 parent: Union[None, 'AbstractFeature', List['AbstractFeature']] = None,
                 root_dir: Union[None, str] = None):
        """
        Args:
            name:
                feature name.
                it is highly recommended that name is unique in one project.
                the name is used as the unique key of dump to local or remote environment.
            parent:
                parent feature. can set List or single Abstract Feature instance.
            root_dir:
                root dir when save experiment logs, object, metrics.
                If string path is set, save data to the directory.
                If None is set, it will look for a root_dir in the parent and if so, it will use that value instead.
                If parent is None too, experiment logs is not stored locally
        """
        self.is_entrypoint = parent is None
        if parent is not None and isinstance(parent, AbstractFeature):
            parent = [parent]
        self.parent = parent

        self._primary_parent = None if parent is None else parent[0]
        self.name = name
        self._root_dir = root_dir

        self.feat_on_train_df = None  # type: Union[None, pd.DataFrame]
        self.feat_on_test_df = None  # type: Union[None, pd.DataFrame]
        self.initialize()

    def __str__(self):
        if self._primary_parent is None:
            return self.name
        return '{}__{}'.format(str(self.name), str(self._primary_parent))

    @property
    def root_dir(self):
        """

        Returns:
            str:
        """
        # この特徴量にディレクトリが指定されているときそれを返す
        if self._root_dir is not None:
            return self._root_dir

        # ディレクトリ指定がなく, もと特徴量も無いとき `None` を返す
        if self.parent is None:
            return None

        # ディレクトリ指定は無いが元特徴量があるとき, そちらに委ねる
        return self._primary_parent.root_dir

    @property
    def has_output_dir(self):
        return self.root_dir is not None

    @property
    def is_recording(self):
        return self.allow_save_local and self.has_output_dir

    @property
    def output_dir(self):
        """
        モデルや特徴量を保存するディレクトリへの path
        Returns:
            str: path string

        """
        if not self.has_output_dir:
            return None
        return os.path.join(self.root_dir, str(self))

    @property
    def dataframe_backend(self):
        return get_dataframe_backend()

    @property
    def output_train_meta_path(self):
        """
        作成した training 時の特徴量 csv への path
        Returns:
            str: path to training csv created by myself

        """
        if not self.has_output_dir:
            return None
        return os.path.join(self.output_dir, self.dataframe_backend.to_filename('train'))

    @property
    def output_test_meta_path(self):
        if not self.has_output_dir: return None
        return os.path.join(self.output_dir, self.dataframe_backend.to_filename('test'))

    @property
    def has_train_meta_path(self) -> bool:
        return self.output_train_meta_path is not None and \
               os.path.exists(self.output_train_meta_path)

    @property
    def has_parent(self) -> bool:
        return self.parent is not None

    @property
    def has_many_parents(self) -> bool:
        return self.has_parent and len(self.parent) > 1

    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        raise NotImplementedError()

    def initialize(self):
        log_output = None
        if self.is_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            log_output = os.path.join(self.output_dir, 'log.txt')

        if self.parent is None:
            logger_name = '****.' + self.name
        else:
            logger_name = self._primary_parent.name + '.' + self.name

        self.logger = get_logger(f'vivid.{logger_name}',
                                 log_level=Settings.LOG_LEVEL,
                                 output_file=log_output,
                                 output_level=Settings.TXT_LOG_LEVEL,
                                 format_str='[vivid.{}] %(message)s'.format(logger_name))

        self.exp_backend = LocalExperimentBackend(output_dir=self.output_dir)

    @contextmanager
    def set_silent(self):
        self.logger.disabled = True
        with self.exp_backend.silent() as exp:
            yield exp
        self.logger.disabled = False

    def fit(self, input_df: pd.DataFrame, y: np.ndarray, force=False) -> pd.DataFrame:
        """
        fit feature to input dataframe

        Args:
            input_df:
                training dataframe.
            y:
                target value
            force:
                force re-fit call. If set `True`, ignore cache train feature and run fit again.

        Returns:
            features corresponding to the training data
        """
        self.initialize()
        if self.has_train_meta_path and not force:
            self.logger.debug('train data is exists. load from local.')
            return self.dataframe_backend.load(self.output_train_meta_path)

        if self.feat_on_train_df is not None:
            return self.feat_on_train_df

        if self.has_parent:
            parent_output_df = pd.DataFrame()
            for parent in self.parent:
                parent_output_df = pd.concat([parent.fit(input_df, y, force=force), parent_output_df], axis=1)
        else:
            parent_output_df = input_df

        with timer(self.logger, format_str='fit: {:.3f}[s]'):
            output_df = self.call(parent_output_df, y, test=False)
            if Settings.CACHE_ON_TRAIN:
                self.feat_on_train_df = output_df

        self.post_fit(input_df, parent_output_df=parent_output_df, out_df=output_df, y=y)
        return output_df

    def post_fit(self,
                 input_df: pd.DataFrame,
                 parent_output_df: pd.DataFrame,
                 out_df: pd.DataFrame,
                 y: np.ndarray):
        """
        a lifecycle method called after fit method.
        To ensure consistency of the output data frame format, the output data frame cannot be modified within this
        function. Therefore, there is no return value.

        If you want to make any changes, please change the call method.

        Args:
            input_df: original input dataframe.
            parent_output_df: dataframe created by parent features.
                Note that it is this value that is passed to call. (Not `input_df`).
                If you calculate the feature importance, usually use `output_df` instead of `input_df`.
            out_df: dataframe created by me.
            y: target.

        Returns:
            Nothing
        """
        if self.is_recording:
            self.dataframe_backend.save(out_df, save_to=self.output_train_meta_path)
        self.exp_backend.mark('train_meta', {
            'shape': out_df.shape,
            'input_columns': input_df.columns,
            'parent_columns': parent_output_df.columns,
            'output_columns': out_df.columns,
            'memory': out_df.memory_usage().sum()
        })
        self.exp_backend.save_object('parent_output_sample', parent_output_df.head(100))
        self.exp_backend.save_object('oof_output_sample', out_df.head(100))

    def predict(self, input_df: pd.DataFrame, recreate=False) -> pd.DataFrame:
        """
        predict new data.

        Notes:
            This method has the ability to cache the return value and is applied when cache_on_test is true in config.
            This is because the features are recursive in structure, preventing the calls from being recursively chained
            in propagation from child to parent.

            Therefore, even if you call `predict` once and then make a prediction on another data, the prediction result
            is not changed. You must explicitly set the recreate argument to True in order to recreate it.

        Args:
            input_df: predict target dataframe
            recreate: optional. If set as `True`, ignore cache file and call core create method (i.e. `self.call`).

        Returns:

        """
        self.initialize()

        if not recreate and self.feat_on_test_df is not None:
            return self.feat_on_test_df
        if self.has_parent:
            output_df = pd.DataFrame()
            for parent in self.parent:
                output_df = pd.concat([parent.predict(input_df, recreate), output_df], axis=1)
        else:
            output_df = input_df
        pred_df = self.call(output_df, test=True)

        if Settings.CACHE_ON_TEST:
            self.feat_on_test_df = pred_df

        if self.is_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            self.dataframe_backend.save(pred_df, self.output_test_meta_path)

        return pred_df
