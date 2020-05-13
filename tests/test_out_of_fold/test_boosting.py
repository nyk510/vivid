import numpy as np
import pytest

from vivid.out_of_fold import boosting
from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks


def test_boosting_seed_block_default_prefix():
    ens = create_boosting_seed_blocks(
        feature_class=boosting.XGBoostClassifierOutOfFold,
        parent=None
    )
    models = [ens, *ens.parent]

    default_prefix = boosting.XGBoostClassifierOutOfFold.__class__.__name__

    for m in models:
        assert default_prefix in m.name


def test_invalid_n_seeds():
    with pytest.raises(ValueError):
        create_boosting_seed_blocks(
            feature_class=boosting.XGBoostClassifierOutOfFold,
            n_seeds=-1
        )


def test_change_seed_number():
    ens = create_boosting_seed_blocks(
        feature_class=boosting.XGBoostClassifierOutOfFold,
        add_init_params={
            'random_state': 0
        },
        n_seeds=10
    )
    models = [ens, *ens.parent]

    seeds = [m._initial_params.get('random_state', None) for m in models if '_ensemble' not in m.name]
    assert len(np.unique(seeds)) == 10, seeds
