import os
import shutil

import pandas as pd
import pytest

from vivid.core import BaseBlock
from vivid.runner import Runner
from .conftest import CounterBlock


def test_block_transform_count(regression_set):
    a = CounterBlock(name='a')
    b = CounterBlock(name='b')
    c = CounterBlock(name='c')

    d = CounterBlock(name='d', parent=[a, b, c])
    e = CounterBlock(name='e', parent=[a, c])
    f = CounterBlock(name='f', parent=[a, d])

    g = CounterBlock(name='g', parent=[e, f])

    input_df, y = regression_set
    runner = Runner(blocks=f)
    runner.fit(input_df, y)

    for block in [a, b, c, d, f]:
        assert block.counter == 1, block

        # 学習時の予測値は取り出せる
        out = runner.load_output(block, is_fit_context=True)
        assert out is not None
        assert isinstance(out, pd.DataFrame)
        assert block.name in out.columns[0]

        # predict はしていないので予測値を取り出そうとするとエラーになる
        with pytest.raises(FileNotFoundError):
            runner.load_output(block, is_fit_context=False)

    for block in [g, e]:
        assert block.counter == 0, block
        with pytest.raises(ValueError):
            runner.load_output(block, is_fit_context=True)

    runner.fit(input_df, y)
    for block in [a, b, c, d, f]:
        assert block.counter == 1, block

    dirpath = os.path.join(runner.experiment.output_dir, d.runtime_env)
    shutil.rmtree(dirpath)

    runner.fit(input_df, y)
    for block in [d, f]:
        assert block.counter == 2, block

    runner.predict(input_df, cache=False)

    for block in [a, b, c, d, f]:
        out = runner.load_output(block, is_fit_context=False)
        assert out is not None
        assert isinstance(out, pd.DataFrame)


def test_re_fit():
    input_df1 = pd.DataFrame({'a': [1, 2, 3]})
    input_df2 = pd.DataFrame({'a': [1, 2, 2]})

    class Count(BaseBlock):
        def fit(self, source_df: pd.DataFrame, y, experiment) -> pd.DataFrame:
            self.vc = source_df['a'].value_counts().to_dict()
            return self.transform(source_df)

        def transform(self, source_df):
            x = source_df['a'].map(self.vc)
            out_df = pd.DataFrame()
            out_df['a_count'] = x.values
            return out_df

    a = Count('a')
    runner = Runner(a)
    runner.fit(input_df1)
    runner.predict(input_df1)
    y_trans = runner.load_output(a, is_fit_context=False)
    print(y_trans)

    runner.fit(input_df2)
    runner.predict(input_df2)
    y_trans2 = runner.load_output(a, is_fit_context=False)
    print(y_trans2)
    assert y_trans.equals(y_trans2)
