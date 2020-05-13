"""Test for core.py files
"""

import os

import pandas as pd
import pytest

from vivid.core import AbstractFeature
from .conftest import SampleFeature


class SimpleMergeFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        return df_source


def test_merge_feature(train_data):
    """特徴量の merge に関するテスト"""
    train_df, y = train_data
    feat1 = SampleFeature()
    feat2 = SampleFeature()
    merged = SimpleMergeFeature(name='merge', parent=[feat1, feat2])
    pred = merged.predict(train_df)

    n_cols = 0
    for feat in [feat1, feat2]:
        p = feat.predict(train_df)
        n_cols += p.shape[1]
    assert pred.shape[1] == n_cols

    assert pred.shape[0] == len(y)
    assert not merged.is_recording


def test_abstract_feature(output_dir):
    abs_entrypoint = AbstractFeature(name='abs1', parent=None)

    model_based = AbstractFeature(name='model1', parent=abs_entrypoint)

    assert abs_entrypoint.is_recording is False
    assert model_based.is_recording is False
    assert model_based.has_train_meta_path is False

    concrete = AbstractFeature(name='concrete', root_dir=output_dir)
    model_concrete = AbstractFeature(name='model1', root_dir=None, parent=concrete)

    assert concrete.is_recording
    assert model_concrete.is_recording

    # model concrete doesn't have his own dir, so the output dir is same as parent dir
    assert output_dir == os.path.dirname(model_concrete.output_dir)

    # set root dir expressly, so this model save his own dir
    model_overthere = AbstractFeature(name='over', root_dir=os.path.join(output_dir, 'the', 'other'),
                                      parent=concrete)
    assert output_dir != os.path.dirname(model_overthere.output_dir)

    class NotSave(AbstractFeature):
        allow_save_local = False

    not_save = NotSave(name='not_save', parent=None, root_dir=output_dir)
    assert not not_save.is_recording


def test_save_train_and_test(train_data, output_dir):
    train_df, y = train_data

    class BasicFeature(AbstractFeature):
        count = 0

        def call(self, df_source: pd.DataFrame, y=None, test=False):
            self.count += 1
            return pd.DataFrame(df_source.iloc[:, 0].values)

    feat = BasicFeature(name='basic', parent=None, root_dir=output_dir)
    assert feat.is_recording
    assert feat.has_train_meta_path is False

    feat.fit(train_df, y)
    assert feat.count == 1
    assert feat.has_train_meta_path

    pred_df = feat.predict(train_df)
    assert feat.count == 2
    assert os.path.exists(feat.output_test_meta_path)

    pred_df2 = feat.predict(train_df.tail(5))
    assert feat.count == 2
    assert len(pred_df2) != 5
    assert len(pred_df2) == len(train_df)

    pred_df3 = feat.predict(train_df.tail(5), recreate=True)
    assert len(pred_df3) == 5
    assert feat.count == 3


@pytest.mark.parametrize('use_cache', [True, False])
def test_features_cache(train_data, use_cache: bool, output_dir):
    """cache を使うかどうかの選択を行えるかどうかのテスト"""
    train_df, y = train_data

    class BasicFeature(AbstractFeature):
        def call(self, df_source: pd.DataFrame, y=None, test=False):
            return pd.DataFrame(df_source.values[:, 0])

    from vivid.env import Settings

    Settings.CACHE_ON_TEST = use_cache
    Settings.CACHE_ON_TRAIN = use_cache

    feat = BasicFeature(name='basic', root_dir=output_dir)

    oof_df = feat.fit(train_df, y)
    test_pred_df = feat.predict(train_df.head(10))

    if use_cache:
        assert feat.feat_on_train_df is not None
        assert feat.feat_on_train_df.equals(oof_df)

        assert feat.feat_on_test_df is not None
        assert feat.feat_on_test_df.equals(test_pred_df)

    else:
        assert feat.feat_on_train_df is None
        assert feat.feat_on_test_df is None
