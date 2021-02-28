"""block runner"""
import gc
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union

import networkx as nx
import pandas as pd
from sklearn.exceptions import NotFittedError
from tabulate import tabulate

from vivid.setup import setup_project
from .backends.experiments import LocalExperimentBackend, ExperimentBackend
from .core import BaseBlock
from .utils import get_logger, timer

logger = get_logger(__name__)


def check_block_fit_output(output_df, input_df):
    if not isinstance(output_df, pd.DataFrame):
        raise ValueError('Block output must be pandas dataframe object. Actually: {}'.format(type(output_df)))


def execute_fit(block, source_df, y, experiment) -> pd.DataFrame:
    with experiment.mark_time('fit'):
        experiment.logger.info(f'start fit {block.name}')
        out_df = block.fit(source_df=source_df,
                           y=y,
                           experiment=experiment)

    check_block_fit_output(out_df, source_df)
    block.report(source_df=source_df,
                 y=y,
                 out_df=out_df,
                 experiment=experiment)

    if experiment.can_save:
        block.frozen(experiment)
        block.clear_fit_cache()
    return out_df


def execute_transform(block, source_df, experiment) -> pd.DataFrame:
    if experiment.can_save:
        # run transform, load data from experiment
        block.unzip(experiment)
    if not block.check_is_fitted(experiment):
        raise NotFittedError(
            'try to execute transform using {} but it is not fitted yet in the current environment. '.format(
                block.name) + \
            'for predict block pass `check_is_fitted` == True after load `unzip` method.'
            ' Check the directory - {} '.format(experiment.namespace)
        )
    out_df = block.transform(source_df)
    return out_df


def sort_blocks(blocks: List[BaseBlock]):
    blocks = set([x for b in blocks for x in b.all_network_blocks()])
    blocks = sorted(list(blocks), key=lambda x: x.name)

    def get_index(b):
        return blocks.index(b)

    G = nx.DiGraph()
    for block in blocks:
        G.add_node(get_index(block), name=block.name)

    for block in blocks:
        for p in block.parent_blocks:
            G.add_edge(
                get_index(p), get_index(block)
            )

    for i in list(nx.topological_sort(G)):
        yield blocks[i]


@dataclass
class EstimatorResult:
    out_df: pd.DataFrame
    block: BaseBlock


def _to_check(done):
    if done:
        return '[x]'
    return '[ ]'


def create_source_df(block, input_df, output_caches, experiment, storage_key, is_fit_context):
    if not block.has_parent:
        return input_df

    source_df = pd.DataFrame()
    for b in block.parent_blocks:
        if b.runtime_env in output_caches:
            _df = output_caches.get(b.runtime_env)
        else:
            with experiment.as_environment(b.runtime_env) as exp:
                _df = b.load_output_from_storage(
                    storage_key=storage_key,
                    experiment=exp,
                    is_fit_context=is_fit_context)
        _df = _df.add_prefix(f'{b.name}__')
        source_df = pd.concat([source_df, _df], axis=1)
    return source_df


@dataclass
class Task:
    order_index: int
    block: BaseBlock
    experiment: LocalExperimentBackend
    done_fit: bool = False
    completed: bool = False
    duration: Union[float, str] = None

    def storage_key(self, is_fit_context: bool):
        return 'train_output' if is_fit_context else 'test_output'

    def __str__(self):
        s = f'- {self.order_index:02d} {_to_check(self.completed)} {_to_check(self.done_fit)} {self.block.name}'
        return s

    def changed_blocks_in_parents(self, tasks: List['Task']) -> List[BaseBlock]:
        retval = []
        for task in tasks:
            if not task.done_fit:
                continue

            if task.block in self.block.parent_blocks:
                retval += [task.block]
        return retval

    def run(self,
            input_df,
            y,
            is_fit_context,
            ignore_cache,
            output_caches):

        block = self.block
        storage_key = self.storage_key(is_fit_context)

        if is_fit_context:
            with self.experiment.as_environment(block.runtime_env) as exp:
                if block.check_is_fitted(exp) and exp.has(storage_key):
                    if not ignore_cache:
                        logger.info(
                            'already fitted and exist output files. use these cache files at {}'.format(exp.output_dir))
                        self.completed = True
                        self.done_fit = False
                        self.duration = 'SKIP'
                        return exp.load_object(storage_key)

                    logger.debug('already exist trained files, but ignore these files. retrain')

        source_df = create_source_df(self.block,
                                     input_df,
                                     output_caches=output_caches,
                                     experiment=self.experiment,
                                     storage_key=storage_key,
                                     is_fit_context=is_fit_context)

        with self.experiment.as_environment(block.runtime_env) as exp:
            timer_name = 'fit' if is_fit_context else 'predict'
            with exp.mark_time(timer_name) as timer:
                if is_fit_context:
                    out_df = execute_fit(block, source_df, y, exp)
                    self.done_fit = True
                else:
                    out_df = execute_transform(block, source_df, exp)
            exp.save_as_python_object(storage_key, out_df)

        self.completed = True
        self.duration = timer.duration
        del source_df
        gc.collect()
        return out_df

    def load_output(self, is_fit_context: bool):
        """block の出力を取り出すための method
        """
        key = self.storage_key(is_fit_context)
        with self.experiment.as_environment(self.block.runtime_env) as exp:
            obj = self.block.load_output_from_storage(storage_key=key,
                                                      experiment=exp,
                                                      is_fit_context=is_fit_context)
            return obj


class Runner:
    """
    Manager class for fit or predict blocks.

    ### Features

    1. order blocks by the dependency so the fit (or transform) method is called only once.
    2. Caches the features once called (unused options available)

    """

    def __init__(self, blocks: Union[BaseBlock, List[BaseBlock]], experiment=None):
        if isinstance(blocks, BaseBlock):
            blocks = [blocks]

        self.blocks = blocks

        if experiment is None:
            import os
            cache = setup_project().cache
            experiment = LocalExperimentBackend(to=os.path.join(cache, 'run__' + str(datetime.now())))
        self.experiment = experiment
        self.tasks = None

    def fit(self,
            train_df,
            y=None,
            cache: bool = True,
            ignore_past_log=False,
            show_each_task=True) -> List[EstimatorResult]:

        estimator_predicts = self._run(
            input_df=train_df,
            y=y,
            cache=cache,
            ignore_cache=ignore_past_log,
            show_each_task=show_each_task,
            is_fit_context=True
        )

        return estimator_predicts

    def predict(self,
                input_df,
                cache: bool = True,
                show_each_task=False) -> List[EstimatorResult]:
        estimator_predicts = self._run(
            input_df=input_df,
            y=None,
            cache=cache,
            is_fit_context=False,
            show_each_task=show_each_task
        )

        return estimator_predicts

    def _initialize(self):
        blocks = list(sort_blocks(self.blocks))
        self.tasks = [Task(i + 1, b, experiment=self.experiment) for i, b in enumerate(blocks)]

    def _run(self,
             input_df,
             y,
             is_fit_context=False,
             cache=True,
             ignore_cache=False,
             show_each_task=False) -> List[EstimatorResult]:
        output_caches = {}
        estimator_predicts = []

        self._initialize()

        root_logger = self.experiment.logger

        root_logger.info('======= start 🚀 ======= ')
        self.show_tasks(self.tasks, is_fit_context)

        for i, task in enumerate(self.tasks):

            changed_blocks = task.changed_blocks_in_parents(self.tasks)
            if len(changed_blocks) > 0:
                root_logger.info('related blocks has changed. so run ignore cache context. / ' +
                                 ','.join(map(str, changed_blocks)))

            with timer(root_logger, prefix=task.block.name + ' '):
                out_df = task.run(input_df=input_df,
                                  y=y,
                                  is_fit_context=is_fit_context,
                                  ignore_cache=ignore_cache or len(changed_blocks) > 0,
                                  output_caches=output_caches)

            if cache:
                output_caches[task.block.runtime_env] = out_df

            if task.block.is_estimator:
                estimator_predicts += [EstimatorResult(out_df=out_df, block=task.block)]

            if show_each_task:
                self.show_tasks(self.tasks, is_fit_context=is_fit_context, output_method=self.experiment.logger.debug)

        root_logger.info('======= Completed!! 🎉 ======= ')
        self.show_tasks(self.tasks, is_fit_context, output_method=self.experiment.logger.info)

        return estimator_predicts

    def show_tasks(self, tasks: List[Task], is_fit_context=False, output_method=print):
        context = 'train' if is_fit_context else 'test'

        def to_dict(task: Task):
            def get_parent_name(block: BaseBlock):
                parents = block.parent_blocks
                n_parents = len(parents)

                s = ','.join([p.name for p in task.block.parent_blocks])
                if len(s) > 40:
                    s = s[:40] + '...'
                if len(s) == 0:
                    s = None

                return f'{n_parents:2d} / {s}'

            return {
                'order': '{:03d}'.format(task.order_index),
                'time[s]': task.duration,
                'done': _to_check(task.completed),
                'fit': _to_check(task.done_fit),
                'name': task.block.name,
                'parent info (N/details)': get_parent_name(task.block)
            }

        data = [to_dict(t) for t in tasks]
        s_metric = tabulate(data, headers='keys', tablefmt='github', floatfmt='.1f', missingval='--')
        for l in s_metric.split('\n'):
            output_method(l)

        output_method('( context={} )'.format(context))
        output_method('-' * 40)

    def load_output(self, block, is_fit_context=False):
        filtered = list(filter(lambda x: x.block == block, self.tasks))
        if len(filtered) == 0:
            raise ValueError('block {} is not defined'.format(block))

        if len(filtered) > 1:
            import warnings
            warnings.warn('duplicate tasks are registered. (tasks expect unique by block.)')
            warnings.warn('use first matched block {}'.format(filtered[0]))

        return filtered[0].load_output(is_fit_context=is_fit_context)


def create_runner(blocks, experiment=None) -> Runner:
    if experiment is None:
        experiment = ExperimentBackend()
    return Runner(blocks, experiment)
