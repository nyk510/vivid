from copy import deepcopy
from typing import Type, Union, List

from vivid.core import AbstractFeature
from vivid.out_of_fold.base import BaseOutOfFoldFeature, EnsembleFeature


def create_boosting_seed_blocks(feature_class: Type[BaseOutOfFoldFeature],
                                parent: Union[None, AbstractFeature, List[AbstractFeature]] = None,
                                prefix: str = None,
                                add_init_params=None,
                                n_seeds=5,) -> EnsembleFeature:
    """
    Boosting Algorithm の seed averaging block を作成する関数.
    feature_class のパラメータに add_init_params & random_state が変わった single model * n_seeds
    + それらのアンサンブル の合計 n_seeds + 1 の配列の特徴量を返す

    Args:
        feature_class(BaseOutOfFoldFeature):
            out of fold feature を継承した, boosting feature.
            boosting 以外では意味をなさないことに注意して下さい.
            (random_state によって seed が代わりそれを averaging することに意味があるアルゴリズムのみが対象です.)
        parent(AbstractFeature):
        prefix(str):
        add_init_params(dict): update init params each averaging feature.
        n_seeds(int): number of averaging feature. must be over zero.

    Returns(List[AbstractFeature]):
    """
    if issubclass(type(feature_class), AbstractFeature):
        raise ValueError(f'invalid `feature_class` argument. `feature_class` must be AbstractFeature subclass.')
    if n_seeds < 0:
        raise ValueError(f'Invalid `n_seeds`. Must be over zero.')
    if prefix is None:
        prefix = str(feature_class.__class__.__name__)
    feats = []
    for seed_id in range(n_seeds):
        add_param = deepcopy(add_init_params)
        if add_param is None:
            add_param = {}
        add_param['random_state'] = seed_id

        feats.append(feature_class(name=f'{prefix}_{seed_id:02d}', parent=parent, add_init_param=add_param))

    ensemble_feat = EnsembleFeature(parent=feats[:], name=f'{prefix}_ensemble')

    return ensemble_feat
