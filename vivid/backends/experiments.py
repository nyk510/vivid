import json
import os
import shutil
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import ContextManager
from typing import Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from vivid.env import get_dataframe_backend
from vivid.json_encoder import NestedEncoder
from vivid.utils import get_logger


class ExperimentBackend:
    """base class for all experiment backends"""

    def __init__(self, namespace=None, logger=None):
        self.namespace = namespace
        self.logger = logger if logger is not None else get_logger(name=__name__)

    def initialize_logger(self):
        pass

    def start(self):
        pass

    def end(self):
        pass

    def has(self, key):
        """whether save object as the key name"""
        return False

    @property
    def can_save(self):
        return self.namespace is not None

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
    def mark_time(self, prefix='') -> ContextManager['ExperimentBackend']:
        start = datetime.now()
        yield self
        end = datetime.now()
        duration = (end - start).total_seconds() / 60
        self.mark(prefix + 'start_at', start)
        self.mark(prefix + 'end_at', end)
        self.mark(prefix + 'duration_minutes', f'{duration:.2f}')

    @contextmanager
    def as_environment(self, *keys, style='flatten') -> ContextManager['ExperimentBackend']:
        yield self

    @contextmanager
    def silent(self):
        _tmp = self.namespace
        self.namespace = None
        self.logger.disabled = True
        yield self
        self.namespace = _tmp
        self.logger.disabled = False

    @property
    def dataframe_backend(self):
        return get_dataframe_backend()

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

    def inner(self: 'LocalExperimentBackend', *args, **kwargs):
        if not self.can_save:
            return None
        return func(self, *args, **kwargs)

    return inner


class LocalExperimentBackend(ExperimentBackend):
    """save object to local storage backend"""

    def __init__(self, namespace=None, mark_filename='metrics.json', keys=None, **kwargs):

        if namespace is not None:
            namespace = os.path.abspath(namespace)
        super(LocalExperimentBackend, self).__init__(namespace, **kwargs)

        self.mark_filename = mark_filename
        self.keys = keys
        if self.can_save:
            os.makedirs(self.namespace, exist_ok=True)

    @property
    def metric_path(self):
        if self.namespace is not None:
            return os.path.join(self.namespace, self.mark_filename)
        return None

    @as_safety
    def has(self, key):
        for n in os.listdir(self.output_dir):
            if key in n:
                return True
        return False

    @property
    def output_dir(self):
        """namespace alias"""
        return self.namespace

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
        return os.path.join(self.namespace, key)

    @as_safety
    def save_json(self, key, obj: dict):
        with open(self._to_path(key, 'json'), 'w') as f:
            json.dump(obj, f, cls=NestedEncoder, indent=4)

    @as_safety
    def save_dataframe(self, key, df: pd.DataFrame):
        df.to_csv(self._to_path(key, 'csv'))

    @as_safety
    def save_figure(self, key, fig: plt.Figure):
        fig.tight_layout()
        fig.savefig(self._to_path(key, 'png'), dpi=120)
        plt.close(fig)

    @as_safety
    def save_as_python_object(self, key, obj):
        p = joblib.dump(obj, self._to_path(key, ext='joblib'))
        self.logger.debug(f'save to {p}')

    @as_safety
    def load_object(self, key):
        return joblib.load(self._to_path(key, ext='joblib'))

    @as_safety
    def clear(self):
        shutil.rmtree(self.output_dir)

    @contextmanager
    def as_environment(self, *keys, style='flatten') -> ContextManager['ExperimentBackend']:
        """
        change environment to the keys and yield return myself


        Args:
            keys:
                new environment's key. list of strings.
        """
        if self.namespace is None:
            namespace = None
        elif style == 'flatten':
            dir_path = os.path.dirname(self.namespace)
            namespace = os.path.join(dir_path, '-'.join(keys))
        else:
            namespace = os.path.join(self.namespace, *keys)
        children = LocalExperimentBackend(namespace=namespace,
                                          mark_filename=self.mark_filename,
                                          logger=self.logger)

        yield children


class CometMLExperimentBackend(ExperimentBackend):
    """expemriment using comet ml. """
