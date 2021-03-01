import pandas as pd
import pytest

from vivid import create_runner
from vivid.backends import LocalExperimentBackend
from vivid.estimators.linear import TunedRidgeBlock
from vivid.features.base import FillnaBlock
from vivid.features.base import get_target_columns


@pytest.fixture
def source_df():
    return pd.DataFrame({
        'a': [1, 2],
        'b': [1, 2]
    })


def test_fillna_block(regression_hasna_set, tmpdir):
    df, y = regression_hasna_set
    runner = create_runner(TunedRidgeBlock(name='ridge', n_trials=1))
    with pytest.raises(ValueError):
        runner.fit(df, y)

    fillna_block = FillnaBlock('FNA')
    ridge = TunedRidgeBlock(parent=fillna_block, name='ridge', n_trials=1)

    runner = create_runner(ridge, experiment=LocalExperimentBackend(tmpdir))
    runner.fit(df, y)

    # can predict re-define blocks (check save require field to tmpdir)
    fillna_block = FillnaBlock('FNA')
    ridge = TunedRidgeBlock(parent=fillna_block, name='ridge', n_trials=1)
    runner = create_runner(ridge, experiment=LocalExperimentBackend(tmpdir))
    runner.predict(df)


def test_target_columns(source_df):
    x = get_target_columns(source_df)
    assert x == ['a', 'b']

    x = get_target_columns(source_df, excludes=['a'])
    assert x == ['b']

    x = get_target_columns(source_df, column=['a'])
    assert x == ['a']

    x = get_target_columns(source_df, column=['a'], excludes=['a'])
    assert x == []


def test_raise_not_exist_column(source_df):
    with pytest.raises(ValueError, match='some specific column does not exist in source_df columns. i.e.'):
        get_target_columns(source_df, column=['a', 'b', 'c'])


def test_raise_not_exist_excludes(source_df):
    with pytest.raises(ValueError, match='some specific `excludes` columns does not exist'):
        get_target_columns(source_df, excludes=['c'])
