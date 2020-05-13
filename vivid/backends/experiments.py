import json
import os
import warnings
from contextlib import contextmanager
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from vivid.json_encoder import NestedEncoder


class ExperimentBackend:
    """base class for all experiment backends"""

    def start(self):
        pass

    def end(self):
        pass

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
    def mark_time(self, prefix=''):
        start = datetime.now()
        yield self
        end = datetime.now()
        duration = (end - start).total_seconds() / 60
        self.mark(prefix + 'start_at', start)
        self.mark(prefix + 'end_at', end)
        self.mark(prefix + 'duration_minutes', f'{duration:.2f}')

    @contextmanager
    def silent(self):
        pass


def save_method(func):
    def inner(self: 'LocalExperimentBackend', *args, **kwargs):
        if not self.has_output_dir:
            return
        return func(self, *args, **kwargs)

    return inner


class LocalExperimentBackend(ExperimentBackend):
    """save object to local storage backend"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.mark_filename = 'metrics.json'

    @property
    def metric_path(self):
        if self.output_dir is not None:
            return os.path.join(self.output_dir, self.mark_filename)
        return None

    @property
    def has_output_dir(self):
        return self.output_dir is not None

    @save_method
    def get_marked(self):
        if os.path.exists(self.metric_path):
            with open(self.metric_path, 'r') as f:
                data = json.load(f)  # type: dict
        else:
            data = {}
        return data

    @save_method
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

    @save_method
    def save_json(self, key, obj: dict):
        with open(self._to_path(key, 'json'), 'w') as f:
            json.dump(obj, f, cls=NestedEncoder, indent=4)

    @save_method
    def save_dataframe(self, key, df: pd.DataFrame):
        df.to_csv(self._to_path(key, 'csv'))

    @save_method
    def save_figure(self, key, fig: plt.Figure):
        fig.tight_layout()
        fig.savefig(self._to_path(key, 'png'), dpi=120)
        plt.close(fig)

    @save_method
    def save_as_python_object(self, key, obj):
        joblib.dump(obj, self._to_path(key, ext='joblib'))

    @contextmanager
    def silent(self):
        _tmp = self.output_dir
        self.output_dir = None
        yield self
        self.output_dir = _tmp


class CometMLExperimentBackend(ExperimentBackend):
    """expemriment using comet ml. """
