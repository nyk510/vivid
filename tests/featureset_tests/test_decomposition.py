import os

import pytest

from vivid import create_runner
from vivid.backends.experiments import LocalExperimentBackend
from vivid.features.decomposition import PCABlock, GaussianMixtureBlock


@pytest.mark.parametrize('block', [
    PCABlock('pca'),
    GaussianMixtureBlock('gmm')
])
def test_pca_block(tmpdir, regression_set, block):
    runner = create_runner(block, experiment=LocalExperimentBackend(to=tmpdir))
    block_dir = os.path.join(tmpdir, block.runtime_env)

    with pytest.raises(FileNotFoundError):
        print(os.listdir(block_dir))

    df, y = regression_set
    runner.fit(train_df=df, y=y)

    for attr in block._save_attributes:
        assert os.path.exists(os.path.join(block_dir, attr + '.joblib')), os.listdir(block_dir)

    runner.predict(df)


def test_add_fit_params(regression_set):
    """`add_fit_params` を変えると clf へ渡す parameter が変化すること"""
    block = PCABlock('pca', add_fit_params={'n_components': 10})
    df, y = regression_set
    runner = create_runner(block)
    runner.fit(df, y)

    assert block.clf_.n_components == 10
