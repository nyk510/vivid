import os
from typing import Dict, Callable

import joblib

from vivid.env import Settings
from vivid.utils import get_logger

logger = get_logger(__name__)


def _get_cache_dir(name, directory=None, extension=None):
    if directory is None:
        directory = Settings.CACHE_DIR
    if extension is None:
        extension = 'joblib'

    path = os.path.join(directory, name + '.' + extension)
    return os.path.abspath(path)


class Wrapper:
    def __init__(self, creator: Callable, save_to):
        if not isinstance(creator, Callable):
            raise ValueError(f'creator must be callable object.')
        self.creator = creator
        self.save_to = save_to

    def __call__(self, *args, **kwargs):
        try:
            return joblib.load(self.save_to)
        except FileNotFoundError:
            pass

        obj = self.creator(*args, **kwargs)
        os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
        joblib.dump(obj, self.save_to)
        return obj


class CacheFunctionFactory:
    wrappers = {}  # type: Dict[str, Wrapper]

    @classmethod
    def generate_wrapper(cls, creator, name, save_dir, extension='joblib'):
        cache_path = _get_cache_dir(name, save_dir, extension)

        w = cls.wrappers.get(name, None)
        if w is not None and w.creator != creator:

            new_name = None
            for i in range(100):
                x = name + '_' + str(i)
                if x not in cls.wrappers.keys():
                    new_name = x
                    break

            if new_name is None:
                raise ValueError('Too many times register as same name. Fix your code.')

            import warnings
            import inspect
            warnings.warn(
                'new registering keyname {} is already exist. Use {} instead.\n'.format(name, new_name) + \
                '{}'.format(inspect.getsource(w.creator)) + \
                '{}'.format(inspect.getsource(creator))
            )
            name = new_name
            cache_path = _get_cache_dir(new_name, save_dir, extension)

        new_wrapper = Wrapper(creator, cache_path)
        cls.wrappers[name] = new_wrapper
        logger.debug('register new function: {} - {}'.format(name, new_wrapper))
        return new_wrapper

    @classmethod
    def list_keys(cls):
        return cls.wrappers.keys()

    @classmethod
    def clear_cache(cls):
        for k, v in cls.wrappers.items():
            logger.info(f'rm {v.save_to}')
            os.remove(v.save_to)


def cacheable(callable_or_scope=None, directory=None):
    """
    make function using cache.
    when decorated by this function, output file save to the cache directory and
    if it is called a second time, the cache file will be used and the decorated method will not be called

    Args:
        callable_or_scope:
        directory:

    Returns:

    """
    if isinstance(callable_or_scope, str):
        def _decorator(create_function=None):
            return CacheFunctionFactory.generate_wrapper(creator=create_function,
                                                         name=callable_or_scope,
                                                         save_dir=directory)

        return _decorator

    if isinstance(callable_or_scope, Callable):
        return CacheFunctionFactory.generate_wrapper(callable_or_scope, name=callable_or_scope.__name__, save_dir=directory)

    raise ValueError(
        'invalid type scape. first argument muse be string (custom fixture name) or callable method {}'.format(
            type(callable_or_scope)))
