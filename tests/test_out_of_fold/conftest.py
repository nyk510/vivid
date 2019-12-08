import pytest

from vivid.env import Settings


@pytest.fixture(scope='session', autouse=True)
def stop_logging():
    before = Settings.LOG_LEVEL
    Settings.LOG_LEVEL = 'WARNING'
    yield
    Settings.LOG_LEVEL = before
