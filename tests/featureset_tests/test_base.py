from vivid.features.base import get_target_columns
import pytest
import pandas as pd


@pytest.fixture
def source_df():
    return pd.DataFrame({
        'a': [1, 2],
        'b': [1, 2]
    })


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
