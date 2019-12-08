"""Test for core.py files
"""

import os

from vivid.core import MergeFeature, EnsembleFeature, AbstractFeature
from .conftest import SampleFeature
from .settings import OUTPUT_DIR


def test_merge_feature(train_data):
    """特徴量の merge に関するテスト"""
    train_df, y = train_data
    feat1 = SampleFeature()
    feat2 = SampleFeature()
    merged = MergeFeature(input_features=[feat1, feat2])
    pred = merged.predict(train_df)

    n_cols = 0
    for feat in [feat1, feat2]:
        p = feat.predict(train_df)
        n_cols += p.shape[1]
    assert pred.shape[1] == n_cols

    assert pred.shape[0] == len(y)
    assert not merged.is_recording


def test_ensemble_feature(train_data):
    train_df, y = train_data
    feat1 = SampleFeature()
    feat2 = SampleFeature()

    ensemble = EnsembleFeature([feat1, feat2])
    pred = ensemble.predict(train_df)
    assert len(train_df) == len(pred)


def test_abstract_feature():
    abs_entrypoint = AbstractFeature(name='abs1', parent=None)

    model_based = AbstractFeature(name='model1', parent=abs_entrypoint)

    assert abs_entrypoint.is_recording is False
    assert model_based.is_recording is False
    assert model_based.has_train_meta_path is False

    parent_dir = OUTPUT_DIR
    concrete = AbstractFeature(name='concrete', root_dir=parent_dir)
    model_concrete = AbstractFeature(name='model1', root_dir=None, parent=concrete)

    assert concrete.is_recording
    assert model_concrete.is_recording

    # model concrete doesn't have his own dir, so the output dir is same as parent dir
    assert parent_dir == os.path.dirname(model_concrete.output_dir)

    # set root dir expressly, so this model save his own dir
    model_overthere = AbstractFeature(name='over', root_dir=os.path.join(OUTPUT_DIR, 'the', 'other'),
                                      parent=concrete)
    assert parent_dir != os.path.dirname(model_overthere.output_dir)

    class NotSave(AbstractFeature):
        allow_save_local = False

    not_save = NotSave(name='not_save', parent=None, root_dir=parent_dir)
    assert not not_save.is_recording
