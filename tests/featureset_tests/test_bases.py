import pytest

from vivid import create_runner
from vivid.backends import LocalExperimentBackend
from vivid.estimators.linear import TunedRidgeBlock
from vivid.features.base import FillnaBlock


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
