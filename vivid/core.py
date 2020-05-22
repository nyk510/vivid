# coding: utf-8
"""
"""

import dataclasses
import gc
import hashlib
from collections import Counter
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd

from .backends.experiments import ExperimentBackend


def network_hash(block: 'BaseBlock') -> str:
    """
    generate unique key of the network structure

    Args:
        block:
            BaseBlock instance.

    Returns:
        unique hash string

    """
    s = block._to_hash()
    for b in block._all_parents():
        s += b._to_hash()
    return hashlib.sha1(s.encode('UTF-8')).hexdigest()[:8] + f'__{block.name}'


@dataclasses.dataclass
class EvaluationEnv:
    """argument dataset pass in report phase. all report functions should be deal this value/"""
    block: 'BaseBlock'
    y: np.ndarray
    output_df: pd.DataFrame
    parent_df: pd.DataFrame
    experiment: ExperimentBackend


class AbstractEvaluation:
    def call(self, env: EvaluationEnv):
        raise NotImplementedError()


class SimpleEvaluation(AbstractEvaluation):
    def call(self, env: EvaluationEnv):
        env.experiment.mark('train_meta', {
            'shape': env.output_df.shape,
            'parent_columns': env.parent_df.columns,
            'output_columns': env.output_df.columns,
            'memory_usage': env.output_df.memory_usage().sum()
        })
        env.experiment.save_object('parent_output_sample', env.parent_df.head(100))
        env.experiment.save_object('oof_output_sample', env.output_df.head(100))


def check_block_fit_output(output_df, input_df):
    if not isinstance(output_df, pd.DataFrame):
        raise ValueError('Block output must be pandas dataframe object. Actually: {}'.format(type(output_df)))


class BaseBlock(object):
    """Abstract class for all feature.

    The feature quantity consists of two main methods.

    * fit: the learning phase, which the internal state is changed based on the input features
    * predict: the prediction phase, which creates a new feature based on the learned state.

    In order to be consistent with the fit/predict conversions, both methods eventually call the call method.
    If you want to create a new feature, override it.

    It is not recommended to override the fit/predict method directly. This is to be consistent with the conversion
    in fit/predict method. We believe that the difference between the code at the time of prediction execution and
    the feature creation code in the learning phase is **the biggest cause of inconsistency** between the training and
    prediction feature.
    """

    # if set True, allow save to local.
    allow_save_local = True
    _forward_cache_key = 'output'
    is_estimator = False

    def __init__(self,
                 name: str,
                 parent: Union[None, 'BaseBlock', List['BaseBlock'], Dict] = None,
                 evaluations: List[AbstractEvaluation] = None):
        """
        Args:
            name:
                feature name.
                it is highly recommended that name is unique in one project.
                the name is used as the unique key of dump to local or remote environment.
            parent:
                parent_blocks feature. can set List or single Abstract Feature instance.
            evaluations:
                report functions. these function must deal the `EvaluationEnv` dataset
                by default, set `SimpleEvaluation`
        """
        self.is_entrypoint = parent is None
        self._parent = parent
        self.name = name
        self._is_root_context = False
        self._refer_cache = 0
        self._cache = {}

        self.primary_block = self._parent[0] if self.has_parent else None
        self._id = None

        if evaluations is None:
            evaluations = [
                SimpleEvaluation()
            ]
        self.evaluations = evaluations

    def _to_hash(self):
        return self.name

    @property
    def runtime_env(self):
        return network_hash(self)

    def check_is_ready_to_predict(self):
        return True

    def __repr__(self):
        return '_'.join([self.name, *[str(m.name) for m in self.parent_blocks]])

    def _all_parents(self) -> List['BaseBlock']:
        return [x for b in self.parent_blocks for x in b._all_parents()] + self.parent_blocks

    @property
    def has_parent(self) -> bool:
        return self._parent is not None

    @property
    def parent_blocks(self) -> List['BaseBlock']:
        """
        get parents as list type.
        If no parents, return empty list `[]`
        If has single parent, converted to list like `[self._parent]`

        Returns:
            array of blocks
        """
        if self._parent is None:
            return []
        if isinstance(self._parent, BaseBlock):
            return [self._parent]
        return self._parent

    def _fit_core(self, source_df, y, experiment) -> pd.DataFrame:
        return self.transform(source_df)

    def transform(self, source_df):
        raise NotImplementedError()

    def frozen(self, experiment: ExperimentBackend):
        """
        save training information to the experiment.

        Args:
            experiment:

        Returns:

        """

        pass

    def unzip(self, experiment: ExperimentBackend):
        """
        load training information from experiment, and ready to prediction phase.
        After call this, the instance expected as ready for transform.
        if you set some attributes (like `self.coef_`) and use it to predict new data transform,
        you should set these attributes

        Args:
            experiment: Experiment Backend
        """
        pass

    def show_network(self, depth=0, prefix=' |-', sep='**'):
        print(f'{sep * depth}{prefix}[{depth}] {self.name}')
        if self.parent_blocks:
            for block in self.parent_blocks:
                block.show_network(depth=depth + 1, prefix=prefix, sep=sep)
        return

    def _save_cache(self, oof_df):
        self._cache[self._forward_cache_key] = oof_df

    def _load_cache(self, logger=None):
        self._refer_cache -= 1
        logger.info(f'{self.name}: remaining refer {self._refer_cache + 1} -> {self._refer_cache}')
        out = self._cache.get(self._forward_cache_key, None)
        if out is None:
            raise ValueError('cant find cache data. ' + ','.join(self._cache.keys()) \
                             + str(self._cache))
        if self._refer_cache == 0 and logger is not None:
            logger.info('clear cache : {}'.format(self.clear_cache()))
        return out

    @property
    def _has_cache_output(self):
        return self._cache.get(self._forward_cache_key, None) is not None

    def _set_network(self):
        nodes = set(self._all_parents())

        # calculate call counts on forward pass
        all_parents = [x for b in nodes for x in b.parent_blocks]

        self._refer_cache = 0
        counts = Counter(all_parents)
        for i, (block, call_count_i) in enumerate(counts.items()):
            block._refer_cache = call_count_i - 1

        self._is_root_context = True
        self._id = 0
        for i, block in enumerate(sorted(counts, key=counts.get)):
            block._id = i + 1
            block._is_root_context = False

    def clear_cache(self):
        del self._cache
        self._cache = {}
        return gc.collect()

    def clear_fit_cache(self):
        """clear data using at fit method.
        It is recommended that in fit method ,
        """
        pass

    def fit(self,
            input_df: pd.DataFrame,
            y: np.ndarray = None,
            experiment: Optional[ExperimentBackend] = None,
            ignore_storage: bool = False,
            root: bool = True) -> pd.DataFrame:
        """
        fit feature to input dataframe

        Args:
            input_df:
                training dataframe.
            y:
                target value
            experiment:
                Experiment object. all objects which have to save are pass to the experiment.
                    (commonly model weight, statistic score)
            ignore_storage:
                force re-fit call. If set `True`, ignore cache files stored at the experiment.
            root:
                regard as root node or not.

        Returns:
            features corresponding to the training data
        """
        return self._forward(input_df, y=y, experiment=experiment, ignore_storage=ignore_storage, root=root,
                             is_fit_context=True)

    def report(self,
               source_df: pd.DataFrame,
               y: np.ndarray,
               out_df: pd.DataFrame,
               experiment: ExperimentBackend = None):
        """
        a lifecycle method called after fit method.
        To ensure consistency of the output data frame format, the output data frame cannot be modified within this
        function. Therefore, there is no return value.

        If you want to make any changes, please change the call method.

        Args:
            source_df:
                dataframe pass to fit_core method.
                In the default behavior, created by parent_blocks features
                Note that it is this value that is passed to call. (Not `input_df`).
                If you calculate the feature importance, usually use `output_df` instead of `input_df`.
            out_df: dataframe created by me. the return value `._fit_core` method.
            y: target.
            experiment: Experiment using at `._fit_core` method.

        Returns:
            Nothing
        """
        env = EvaluationEnv(block=self, output_df=out_df, parent_df=source_df, experiment=experiment, y=y)

        for callback in self.evaluations:
            try:
                callback.call(env)
            except Exception as e:
                import warnings
                warnings.warn(str(e))

    def predict(self,
                input_df: pd.DataFrame,
                experiment: Optional[ExperimentBackend] = None,
                ignore_storage: bool = False,
                root: bool = True) -> pd.DataFrame:
        """
        predict new data.

        Notes:
            This method has the ability to cache the return value and is applied when cache_on_test is true in config.
            This is because the features are recursive in structure, preventing the calls from being recursively chained
            in propagation from child to parent_blocks.

            Therefore, even if you call `predict` once and then make a prediction on another data, the prediction result
            is not changed. You must explicitly set the recreate argument to True in order to recreate it.

        Args:
            input_df: predict target dataframe
            ignore_storage: optional. If set as `True`, ignore cache file and call core create method (i.e. `self.call`).

        Returns:
            predict data
        """
        return self._forward(input_df, None, experiment, root, ignore_storage, is_fit_context=False)

    def _before_forward(self, root: bool):
        if root:
            self._set_network()
            self.clear_cache()
        return self

    def _forward(self, input_df: pd.DataFrame, y: Union[None, np.ndarray],
                 experiment: Optional[ExperimentBackend] = None, root: bool = False,
                 ignore_storage: bool = False, is_fit_context: bool = False):
        self._before_forward(root=root)
        if experiment is None:
            experiment = ExperimentBackend()

        if not isinstance(input_df, pd.DataFrame):
            raise ValueError('Invalid `input_df`. Must be pandas DataFrame object. Actually: {}'.format(type(input_df)))
        if self._has_cache_output:
            return self._load_cache(experiment.logger)

        with experiment.as_environment(self.runtime_env,
                                       style='nested' if root else 'flatten') as my_exp:
            out_df = self._create_or_load_output(input_df,
                                                 y=y,
                                                 experiment=my_exp,
                                                 recreate=ignore_storage,
                                                 is_fit_context=is_fit_context)

        if self._refer_cache > 0:
            self._save_cache(out_df)
        self._after_forward(output_df=out_df, root=root, experiment=experiment)
        return out_df

    def _after_forward(self, output_df, root, experiment):
        if self._refer_cache > 0:
            self.feat_on_train_df = output_df
        if root:
            experiment.logger.info('clear cache : {}'.format(self.clear_cache()))
        return self

    def _create_or_load_output(self,
                               input_df,
                               y,
                               experiment,
                               recreate,
                               is_fit_context: bool = False) -> pd.DataFrame:
        storage_key = 'train_oof' if is_fit_context else 'new_predict'

        if not recreate and experiment.has(storage_key):
            try:
                # if there is a meta-file on experiment, load from experiment, not create
                experiment.logger.info('try: load from storage')
                out_df = experiment.load_object(storage_key)
                experiment.logger.info('success.')
                return out_df
            except FileNotFoundError as e:
                experiment.logger.warning(e)

        parent_out_df = self._prepare_parent_output(input_df, y, experiment, recreate, is_fit_context=is_fit_context)
        if is_fit_context:
            with experiment.mark_time('train_'):
                experiment.logger.info(f'start train {self.name}')
                out_df = self._fit_core(source_df=parent_out_df, y=y, experiment=experiment)

            check_block_fit_output(out_df, parent_out_df)
            self.report(source_df=parent_out_df, y=y, out_df=out_df, experiment=experiment)

            if experiment.can_save:
                self.frozen(experiment)
                self.clear_fit_cache()
        else:
            if experiment.can_save:
                # run transform, store as local attribute
                self.unzip(experiment)
            self.check_is_ready_to_predict()
            out_df = self.transform(parent_out_df)

        experiment.save_as_python_object(storage_key, out_df)
        del parent_out_df
        gc.collect()
        return out_df

    def _prepare_parent_output(self,
                               input_df: pd.DataFrame,
                               y=None,
                               experiment: ExperimentBackend = None,
                               recreate: bool = False,
                               is_fit_context=False) -> pd.DataFrame:
        # prepare parent outputs
        parent_out_df = pd.DataFrame()
        if self.has_parent:
            for b in self.parent_blocks:
                if is_fit_context:
                    out_i = b.fit(input_df, y=y, experiment=experiment, ignore_storage=recreate, root=False)
                else:
                    out_i = b.predict(input_df, experiment, ignore_storage=recreate, root=False, )

                out_i = out_i.add_prefix(b.name + '_')
                parent_out_df = pd.concat([parent_out_df, out_i], axis=1)
        else:
            parent_out_df = input_df

        return parent_out_df
