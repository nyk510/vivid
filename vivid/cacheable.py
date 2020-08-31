import inspect
import os
import shutil
from typing import Dict, Callable

import joblib

from vivid.setup import setup_project
from vivid.utils import get_logger
from vivid.utils import param_to_name

logger = get_logger(__name__)


def to_hashable_dict(kwrgs: dict):
    retval = {}
    for k, v in kwrgs.items():
        try:
            retval[k] = hash(v)
        except TypeError as e:
            logger.warn(f'pass none hashable object at key={k}. ')
            logger.warn(e)
    return frozenset(retval.items())


class CacheFunction:
    def __init__(self, create_function: Callable, save_dir):
        if not isinstance(create_function, Callable):
            raise ValueError(f'Create function must be callable object.')
        self.create_function = create_function
        self.save_dir = save_dir

    def get_identify_string(self, params: dict):
        """
        params に一意な文字列を取得する

        Args:
            params:
                文字列化する parameter の dict.
        """
        # note:
        # hash 関数の値は session 間で変動するため, parameter を string 評価した時の値としている.
        # しかしこの方法だと key / value のいずれかに object 等 string として評価したくない値が来た時に困る.
        # (例えば滅茶苦茶文字列が長くなる, 等)
        # 将来的には hash 化した値で以前の parameter と一致しているかどうかを判断するようにしたい.
        s = param_to_name(params)
        if len(s) == 0:
            s = 'default'
        return s

    def __call__(self, *args, **kwargs):
        os.makedirs(self.save_dir, exist_ok=True)

        call_args = inspect.getcallargs(self.create_function, *args, **kwargs)
        identify_args = self.get_identify_string(call_args)

        save_to = os.path.join(self.save_dir, str(identify_args) + '.joblib')

        try:
            logger.info('load from {}'.format(save_to))
            return joblib.load(save_to)
        except FileNotFoundError:
            logger.info(f'file is not found. create newly.')
            pass

        obj = self.create_function(*args, **kwargs)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        joblib.dump(obj, save_to)
        logger.info('create {} is competed ;)'.format(save_to))
        return obj


class CacheFunctionFactory:
    wrappers = {}  # type: Dict[str, CacheFunction]

    @classmethod
    def generate_cache_function(cls, creator, name, save_dir) -> CacheFunction:
        if cls.is_registered(name):
            registered = cls.get_cache_function(name)
            if registered.create_function != creator:

                new_name = None
                for i in range(100):
                    x = name + '_' + str(i)
                    if x not in cls.wrappers.keys():
                        new_name = x
                        break

                if new_name is None:
                    raise ValueError('Too many times register as same name. Fix your code.')

                import warnings
                warnings.warn(
                    'new registering keyname {} is already exist. Use {} instead.\n'.format(name, new_name) + \
                    '{}'.format(inspect.getsource(registered.create_function)) + \
                    '{}'.format(inspect.getsource(creator))
                )
                name = new_name

        new_wrapper = CacheFunction(creator, save_dir=os.path.join(save_dir, name))
        cls.wrappers[name] = new_wrapper
        logger.debug('register new function: {} - {}'.format(name, new_wrapper))
        return new_wrapper

    @classmethod
    def list_keys(cls):
        return cls.wrappers.keys()

    @classmethod
    def is_registered(cls, key):
        return key in cls.wrappers

    @classmethod
    def get_cache_function(cls, key):
        if not cls.is_registered(key):
            return None
        return cls.wrappers.get(key)

    @classmethod
    def function_is_registered(cls, func):
        return func in cls.wrappers.values()

    @classmethod
    def clear_cache(cls, key):
        if not cls.is_registered(key):
            return

        cache_func = cls.get_cache_function(key)
        shutil.rmtree(cache_func.save_dir)
        logger.debug('rm all cache files at {}'.format(cache_func.save_dir))


def cacheable(callable_or_scope=None, directory=None):
    """
    make function using cache.
    It's useful for functions that take a lot of time to process

    when decorated by this function, output file save to the cache directory and
    if it is called a second time, the cache file will be used and the decorated method will not be called

    Args:
        callable_or_scope:
            create function or scope (i.e. save name).
        directory:
            path to the directory where the cache file is stored.
            By default, use `cache` attr in the return value of `setup_project`

    Returns:
        cache enabled function
    """
    if directory is None:
        directory = setup_project().cache

    if isinstance(callable_or_scope, str):
        def _decorator(create_function=None):
            return CacheFunctionFactory.generate_cache_function(creator=create_function,
                                                                name=callable_or_scope,
                                                                save_dir=directory)

        return _decorator

    if isinstance(callable_or_scope, Callable):
        return CacheFunctionFactory.generate_cache_function(callable_or_scope,
                                                            name=callable_or_scope.__name__,
                                                            save_dir=directory)

    raise ValueError(
        'invalid type scape. first argument muse be string (custom fixture name) or callable method {}'.format(
            type(callable_or_scope)))
