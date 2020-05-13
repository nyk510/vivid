import pytest

from vivid.backends.dataframes import CSVBackend, JoblibBackend
from vivid.env import get_dataframe_backend, Settings


class CustomBackend:
    pass


def test_load_from_cache():
    backend = get_dataframe_backend()
    backend2 = get_dataframe_backend()
    assert backend == backend2
    assert backend == get_dataframe_backend.backend


@pytest.mark.parametrize('name,expected_class', [
    ('vivid.backends.dataframes.CSVBackend', CSVBackend),
    ('vivid.backends.dataframes.JoblibBackend', JoblibBackend),
    ('tests.test_env.CustomBackend', CustomBackend)  # can import from string case
])
def test_load_csv_backend(name, expected_class):
    """can load specific backend"""
    backend = get_dataframe_backend(name)

    assert isinstance(backend, expected_class)


@pytest.mark.parametrize('name,expected_class', [
    ('vivid.backends.dataframes.CSVBackend', CSVBackend),
    ('vivid.backends.dataframes.JoblibBackend', JoblibBackend),
    ('tests.test_env.CustomBackend', CustomBackend)  # can import from string case
])
def test_can_change_default_backend(name, expected_class):
    Settings.DATAFRAME_BACKEND = name
    backend = get_dataframe_backend()
    assert isinstance(backend, expected_class)


def test_fail_load_not_exist_name():
    with pytest.raises(ImportError):
        get_dataframe_backend('foo')
