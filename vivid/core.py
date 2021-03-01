# coding: utf-8
"""
"""

import dataclasses
import gc
import hashlib
from typing import Union, List
import warnings
import numpy as np
import pandas as pd

from .backends.experiments import ExperimentBackend
from .utils import get_logger

logger = get_logger(__name__)


def network_hash(block: 'BaseBlock', size=8) -> str:
    """
    generate unique key of the network structure.

    Args:
        block:
            BaseBlock instance.
        size:
            hash key string length. default=8.

    Returns:
        unique hash string

    """
    s = block._to_hash()

    for b in block.parent_blocks:
        s += network_hash(b, size=size)
    return block.name + '__' + hashlib.sha1(s.encode('UTF-8')).hexdigest()[:size]


def not_fitted_blocks(block: 'BaseBlock', experiment: ExperimentBackend) -> List['BaseBlock']:
    """
    whether the all network related to the block in the experiment

    Args:
        block:
        experiment:

    Returns:

    """
    retval = []
    if not block.check_is_fitted(experiment):
        experiment.logger.info(f'> [ ] {block.name}: NG: not fitted.')
        retval += [block]
    else:
        experiment.logger.info(f'> [x]: {block.name} OK')

    if block.has_parent:
        experiment.logger.info(f'request from {block.name} ---')
    for parent in block.parent_blocks:
        with experiment.as_environment(parent.runtime_env, style='flatten') as exp:
            retval += not_fitted_blocks(parent, exp)
    return retval


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


def to_statistic(input_df: pd.DataFrame) -> dict:
    """
    データフレームの基礎統計量の算出

    Args:
        input_df:
            target dataframe

    Returns:
        dict which has statistic data
    """
    return {
        'shape': input_df.shape,
        'null': input_df.isnull().sum(),
        'memory_usage': input_df.memory_usage().sum(),
        'columns': input_df.keys()
    }


class SimpleEvaluation(AbstractEvaluation):
    def call(self, env: EvaluationEnv):
        meta = {}
        data = {
            'input': env.parent_df,
            'output': env.output_df
        }

        for k, df in data.items():
            meta[k] = to_statistic(df)

        env.experiment.mark('train_meta', meta)
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

    It is not recommended to override the fit/predict method directly.
    This is to be consistent with the conversion in fit/predict method.

    We believe that the difference between the code at the time of prediction execution and the feature creation code in
    the learning phase is **the biggest cause of inconsistency** between the training and prediction feature.
    """

    # if set True, allow save to local.
    allow_save_local = True
    is_estimator = False

    # save attributes in frozen / unzip method.
    # set these fields which calculate and set in `fit` method and use in `transform` method.
    _save_attributes = []

    def __init__(self,
                 name: str,
                 parent: Union[None, 'BaseBlock', List['BaseBlock']] = None,
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
                evaluation functions. these function must be callable, and deal the `EvaluationEnv` dataset.
                it is recommended that the sub-class of `AbstractEvaluation` class.
                by default, set `SimpleEvaluation`.
        """
        self.is_entrypoint = parent is None
        self._parent = parent
        self.name = name
        self._is_root_context = False
        self._refer_cache = 0
        self._cache = {}
        self._id = None

        if evaluations is None:
            evaluations = [
                SimpleEvaluation()
            ]
        self.evaluations = evaluations

    def _to_hash(self) -> str:
        """
        describe the block uniquely as possible

        Returns:
            a String which is almost unique in blocks
        """
        return self.name

    @property
    def runtime_env(self):
        """block を保存する際につけるディレクトリの命名.
        [NOTE]: 名前が一致しているかつ, parent が異なる block 同士での名前の batting を避けるために hash 化処理を入れていたが,
        実用上名前がバッティングすることはほぼ無い
        """
        return network_hash(self)

    def check_is_fitted(self, experiment: ExperimentBackend) -> bool:
        """
        whether this block is `fitted` (i.e. ready to predict new data)

        Args:
            experiment:
                Current Experiment.

        Returns:
            boolean. return True, this block is ready to predict
        """
        return True

    def all_network_blocks(self):
        retval = [self]
        for b in self.parent_blocks:
            retval += b.all_network_blocks()
        return retval

    def __repr__(self):
        return self.__class__.__name__ + '__' + '_'.join([self.name, *[str(m.name) for m in self.parent_blocks]])

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
            array of parent blocks
        """
        if self._parent is None:
            return []
        if isinstance(self._parent, BaseBlock):
            return [self._parent]
        return self._parent

    def fit(self,
            source_df: pd.DataFrame,
            y: Union[None, np.ndarray],
            experiment: ExperimentBackend) -> pd.DataFrame:
        """Converts the internal state to match the training data.

        NOTE:
            If you are only doing a transformation (e.g., one that does not depend on training data, such as min-max scaling),
            do not do anything inside this function and implement the transformation logic in transform.
            It is based on the idea that you want to use the same code for prediction and training transformations.

        Args:
            source_df:
                source dataframe for fitting. pandas DataFrame object.
            y: target numpy array.
            experiment:
                Current Experiment

        Returns:
            feature data that correspond to source feature.
            In the case of machine learning models, the return value is an out of fold prediction.
        """

        return self.transform(source_df)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        transform new data.

        Args:
            source_df:
                input data frame. pandas dataframe object

        Returns:
            feature data that correspond to source feature.
            In the case of machine learning models, the return value is prediction values.
        """
        raise NotImplementedError()

    def frozen(self, experiment: ExperimentBackend) -> 'BaseBlock':
        """
        save training information to the experiment.

        Args:
            experiment: Experiment Backend

        Returns:

        """
        for field in self._save_attributes:
            try:
                object = getattr(self, field)
            except AttributeError:
                warnings.warn(f'field `{field}` is not found in the block `{self.name}`. '
                               'Check your implementation, usually typo or mistakes at `_save_attributes`')
                continue
            experiment.save_as_python_object(field, object)
        return self

    def unzip(self, experiment: ExperimentBackend):
        """
        load training information from experiment, and ready to prediction phase.
        After call this, the instance expected as ready for transform.
        if you set some attributes (like `self.coef_`) and use it to predict new data transform,
        you should set these attributes

        Args:
            experiment: Experiment Backend
        """
        for field in self._save_attributes:
            try:
                object = experiment.load_object(field)
            except FileNotFoundError:
                warnings.warn(
                    f'Fail to load `{field}`. It is not found in current experiment context ;( . '
                    f'If you use the field in `transform` method, it doesnt work well (missing attribute error)')
                continue
            setattr(self, field, object)

    def clear_fit_cache(self):
        gc.collect()

    def show_network(self, depth=0, prefix=' |-', sep='**'):
        print(f'{sep * depth}{prefix}[{depth}] {self.name} <{self.runtime_env}>')
        if self.parent_blocks:
            for block in self.parent_blocks:
                block.show_network(depth=depth + 1, prefix=prefix, sep=sep)
        return

    def report(self,
               source_df: pd.DataFrame,
               out_df: pd.DataFrame,
               y: np.ndarray,
               experiment: ExperimentBackend = None):
        """
        a lifecycle method called after fit method.
        To ensure consistency of the output data frame format, the output data frame cannot be modified within this
        function.

        Args:
            source_df:
                dataframe pass to fit_core method.
                In the default behavior, created by parent_blocks features
                Note that it is this value that is passed to call. (Not `input_df`).
                If you calculate the feature importance, usually use `output_df` instead of `input_df`.
            out_df:
                dataframe created by myself (i.e. the return value created by `.fit` method.)
            y:
                target
            experiment:
                Experiment using at `.fit` method.

        Returns:
            Nothing
        """
        env = EvaluationEnv(block=self, output_df=out_df, parent_df=source_df, experiment=experiment, y=y)

        for callback in self.evaluations:
            experiment.logger.debug('start evaluation: {}'.format(callback))
            try:
                callback.call(env)
            except Exception as e:
                import warnings
                warnings.warn('raise Exception on {}. '.format(callback) + str(e))

    def load_output_from_storage(self,
                                 storage_key: str,
                                 experiment: ExperimentBackend,
                                 is_fit_context=False) -> pd.DataFrame:
        if not experiment.has(storage_key):
            raise FileNotFoundError('{} is not found at experiment: {}'.format(storage_key, experiment))

        if is_fit_context:
            not_fit_blocks = not_fitted_blocks(self, experiment)
            if len(not_fit_blocks) > 0:
                raise FileNotFoundError(
                    'Network has not fitted blocks. ' + ', '.join([b.name for b in not_fit_blocks]) + \
                    '. create these blocks and renew stored output features.')

            logger.info('All Feature are ready to prediction. Load cache output.')

            # if there is a meta-file on experiment, load from experiment, not create
        experiment.logger.info('load from storage: {}'.format(storage_key))
        output = experiment.load_object(storage_key)
        return output
