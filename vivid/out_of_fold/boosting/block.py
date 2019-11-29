from copy import deepcopy
from typing import Type

from vivid.core import AbstractFeature, EnsembleFeature
from vivid.out_of_fold.base import BaseOutOfFoldFeature


def create_boosting_seed_blocks(parent: AbstractFeature,
                                prefix: str,
                                feature_class: Type[BaseOutOfFoldFeature],
                                add_init_params=None,
                                n_seeds=5):
    """
    Boosting Algorithm の seed averaging block を作成する関数.
    feature_class のパラメータに add_init_params & random_state が変わった single model * n_seeds
    + それらのアンサンブル の合計 n_seeds + 1 の配列の特徴量を返す

    Args:
        parent(AbstractFeature):
        prefix(str):
        feature_class(BaseOutOfFoldFeature):
            out of fold feature を継承した, boosting feature.
            boosting 以外では意味をなさないことに注意して下さい.
            (random_state によって seed が代わりそれを averaging することに意味があるアルゴリズムのみが対象です.)
        add_init_params(dict):

    Returns(List[AbstractFeature]:
    """
    if issubclass(type(feature_class), AbstractFeature):
        raise ValueError('invalid feature class. must set AbstractFeature class.')

    feats = []
    for seed_id in range(n_seeds):
        add_param = deepcopy(add_init_params)
        if add_param is None:
            add_param = {}
        add_param['random_state'] = seed_id

        feats.append(feature_class(name=f'{prefix}_{seed_id:02d}', parent=parent, add_init_param=add_param))

    if n_seeds > 1:
        ensemble_feat = EnsembleFeature(feats[:], name=f'{prefix}_ensemble', root_dir=parent.root_dir)
        feats.append(ensemble_feat)
    return feats
