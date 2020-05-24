import pandas as pd
import pytest

from vivid.core import BaseBlock, network_hash


def test_network_hash():
    a = BaseBlock('a')
    b = BaseBlock('b')
    assert network_hash(a) != network_hash(b)
    assert network_hash(a) == network_hash(a)

    c = BaseBlock('c', parent=[a, b])
    hash1 = network_hash(c)
    a._parent = [BaseBlock('z')]
    hash2 = network_hash(c)
    assert hash1 != hash2


class CounterBlock(BaseBlock):
    def __init__(self, **kwargs):
        super(CounterBlock, self).__init__(**kwargs)
        self.counter = 0

    def _fit_core(self, source_df, y, experiment) -> pd.DataFrame:
        self.counter += 1
        return source_df.copy()


def test_invalid_argument():
    a = CounterBlock(name='a')
    x = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        a.fit(x)


def test_collect_parent():
    a = BaseBlock('a')
    b = BaseBlock('b')
    c = BaseBlock('c', parent=[a, b])
    d = BaseBlock('d', parent=[c, a])

    parent = d._all_parents()
    assert sorted(parent, key=lambda x: x.name) == sorted([a, b, c, a], key=lambda x: x.name)


def test_invalid_fit_core_implement(regression_set):
    class A(BaseBlock):
        def _fit_core(self, source_df, y, experiment) -> pd.DataFrame:
            return experiment

    a = A(name='a')
    input_df, y = regression_set

    with pytest.raises(ValueError):
        a.fit(input_df, y)


def test_block_transform_count(regression_set):
    a = CounterBlock(name='a')
    b = CounterBlock(name='b')
    c = CounterBlock(name='c')

    d = CounterBlock(name='d', parent=[a, b, c])
    e = CounterBlock(name='e', parent=[a, c])
    f = CounterBlock(name='f', parent=[a, d])

    g = CounterBlock(name='g', parent=[e, f])

    input_df, y = regression_set
    g.fit(input_df, y)

    assert g._is_root_context, g
    for block in [a, b, c, d, e, f]:
        assert block.counter == 1, block
        assert not block._is_root_context
        assert len(block._cache.keys()) == 0

    e.fit(input_df, y)
    assert e._is_root_context, e


def test_re_fit():
    input_df1 = pd.DataFrame({'a': [1, 2, 3]})
    input_df2 = pd.DataFrame({'a': [1, 2, 2]})

    class Count(BaseBlock):
        def _fit_core(self, source_df: pd.DataFrame, y, experiment) -> pd.DataFrame:
            self.vc = source_df['a'].value_counts().to_dict()
            return self.transform(source_df)

        def transform(self, source_df):
            x = source_df['a'].map(self.vc)
            out_df = pd.DataFrame()
            out_df['a_count'] = x.values
            return out_df

    a = Count('a')
    a.fit(input_df1)

    y_trans = a.predict(input_df1)

    a.fit(input_df2)
    y_trans2 = a.predict(input_df2)
    assert not y_trans.equals(y_trans2)
