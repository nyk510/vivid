import json
import os
import shutil
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import ContextManager
from typing import Union, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from vivid.env import get_dataframe_backend
from vivid.json_encoder import NestedEncoder
from vivid.utils import get_logger
from vivid.utils import Timer


class ExperimentBackend:
    """base class for all experiment backends"""

    def __init__(self,
                 to=None,
                 keys: Union[List[str], None] = None,
                 logger=None,
                 datafame_backend=None):
        self.to = to
        self.keys = [] if keys is None else keys
        self.logger = get_logger(__name__)
        self.dataframe_backend = get_dataframe_backend() if datafame_backend is None else datafame_backend

    def __enter__(self):
        self.start()

    def start(self):
        pass

    def end(self):
        pass

    def has(self, key):
        """whether save object as the key name"""
        return False

    @property
    def can_save(self):
        return self.to is not None

    def save_object(self, key, obj):
        mapping = {
            pd.DataFrame: self.save_dataframe,
            plt.Figure: self.save_figure,
            dict: self.save_json,
            str: self.mark
        }

        func = self.save_as_python_object

        for cls_type, val in mapping.items():
            if isinstance(obj, cls_type):
                func = val
                break

        try:
            func(key, obj)
        except Exception as e:
            warnings.warn(f'Error has occurred when save experiment object {key} {type(obj)}')
            warnings.warn(str(e))
            return

    def load_object(self, key):
        raise FileNotFoundError()

    def get_marked(self) -> Union[None, dict]:
        return None

    def mark(self, key, value):
        pass

    def save_json(self, key, obj: dict):
        pass

    def save_dataframe(self, key, df: pd.DataFrame):
        pass

    def save_figure(self, key, fig: plt.Figure):
        pass

    def save_as_python_object(self, key, obj):
        pass

    @contextmanager
    def mark_time(self, prefix: str) -> ContextManager[Timer]:
        timer = Timer(logger=self.logger, prefix=prefix)
        with timer:
            yield timer
        self.mark(prefix, {
            'start_at': timer.start,
            'end_at': timer.end,
            'duration_minutes': '{:.2f}'.format(timer.duration)
        })

    @contextmanager
    def as_environment(self, *keys, style='flatten') -> ContextManager['ExperimentBackend']:
        yield self

    @contextmanager
    def silent(self):
        _tmp = self.to
        self.to = None
        self.logger.disabled = True
        yield self
        self.to = _tmp
        self.logger.disabled = False

    def clear(self):
        pass

    def set_silent(self):
        self.logger.disabled = True


def as_safety(func):
    """
    If the instance has not output directory, pass call method.

    Args:
        func:
            decorate function.
            expect the method of a class that has `has_output_dir` attribute.

    Returns:
        wrapper function
    """

    def inner(self: 'ExperimentBackend', *args, **kwargs):
        if not self.can_save:
            self.logger.info('cant allow save. skip call.')
            return None
        return func(self, *args, **kwargs)

    return inner


def get_logger_name(to, keys: List[str]):
    prepends = []
    if to:
        prepends += [os.path.basename(to)]
    else:
        prepends += [__name__]
    prepends += map(str, keys)
    logger_name = '/'.join(prepends)
    return logger_name


class LocalExperimentBackend(ExperimentBackend):
    """save object to local storage backend"""

    def __init__(self, to=None, mark_filename='metrics.json', **kwargs):
        if to is not None:
            to = os.path.abspath(to)
        super(LocalExperimentBackend, self).__init__(to, **kwargs)

        self.mark_filename = mark_filename

        if self.can_save:
            os.makedirs(self.output_dir, exist_ok=True)

        logger_name = get_logger_name(to, keys=self.keys)
        self.logger = get_logger(name=logger_name,
                                 output_file=self.logging_path,
                                 format_str='%(asctime)s: %(name)-10s: %(message)s')
        self.logger.debug('experiment output is {}'.format(self.output_dir))
        self.logger.debug('logger name: {}'.format(logger_name))

    @property
    def logging_path(self):
        if not self.output_dir:
            return None

        return os.path.join(self.output_dir, 'log.txt')

    @property
    def metric_path(self):
        if self.to is not None:
            return os.path.join(self.output_dir, self.mark_filename)
        return None

    @as_safety
    def has(self, key):
        for n in os.listdir(self.output_dir):
            if key in n:
                return True
        return False

    @property
    def output_dir(self):
        """exact output directory"""
        if self.to is None:
            return None
        if self.keys is None:
            return self.to
        return os.path.join(self.to, *self.keys)

    @as_safety
    def get_marked(self):
        if os.path.exists(self.metric_path):
            with open(self.metric_path, 'r') as f:
                data = json.load(f)  # type: dict
        else:
            data = {}
        return data

    @as_safety
    def mark(self, key, value):
        update_data = {
            key: value
        }
        data = self.get_marked()
        data.update(update_data)
        return self.save_json(self.mark_filename, data)

    def _to_path(self, key, ext):
        if '.' in key:
            key = key
        else:
            key = f'{key}.{ext}'
        return os.path.join(self.output_dir, key)

    @as_safety
    def save_json(self, key, obj: dict, indent=2, ensure_ascii=False, **kwargs):
        with open(self._to_path(key, 'json'), 'w') as f:
            json.dump(obj, f, cls=NestedEncoder, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

    @as_safety
    def save_dataframe(self, key, df: pd.DataFrame, **kwargs):
        df.to_csv(self._to_path(key, 'csv'), **kwargs)

    @as_safety
    def save_figure(self, key, fig: plt.Figure):
        fig.tight_layout()
        fig.savefig(self._to_path(key, 'png'), dpi=120)
        plt.close(fig)

    @as_safety
    def save_as_python_object(self, key, obj):
        p = joblib.dump(obj, self._to_path(key, ext='joblib'))
        self.logger.debug(f'save {str(obj)} to {p}')

    @as_safety
    def load_object(self, key):
        return joblib.load(self._to_path(key, ext='joblib'))

    @as_safety
    def clear(self):
        shutil.rmtree(self.output_dir)

    @contextmanager
    def as_environment(self, *keys, style='nested') -> ContextManager['LocalExperimentBackend']:
        """
        change environment to the keys and yield return myself

        Args:
            keys:
                new environment's key. list of strings.
        """
        if style == 'flatten':
            keys = list(keys)
        elif style == 'nested':
            keys = self.keys + list(keys)
        else:
            raise ValueError('style is must be {}. Actually, {}'.format(','.join(['flatten', 'nested']),
                                                                        style))

        children = LocalExperimentBackend(to=self.to,
                                          mark_filename=self.mark_filename,
                                          keys=keys,
                                          datafame_backend=self.dataframe_backend)

        if self.logger.disabled:
            with children.silent():
                yield children
        else:
            yield children

        del children


class CometMLExperimentBackend(ExperimentBackend):
    """expemriment using comet ml. """
