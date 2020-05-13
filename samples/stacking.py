from itertools import product

import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import AbstractFeature
from vivid.out_of_fold import EnsembleFeature
from vivid.out_of_fold.boosting import XGBoostRegressorOutOfFold, OptunaXGBRegressionOutOfFold, LGBMRegressorOutOfFold
from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold
from vivid.out_of_fold.kneighbor import KNeighborRegressorOutOfFold
from vivid.out_of_fold.linear import RidgeOutOfFold


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        out_df = pd.DataFrame()
        n_cols = len(df_source.columns)
        for a, b in product(range(n_cols), range(n_cols)):
            out_df[f'sum_{a}_{b}'] = df_source[a] + df_source[b]
        return out_df


class CopyFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    copy_feat = CopyFeature(name='copy', root_dir='./boston_stacking')
    process_feat = BostonProcessFeature(name='boston_base', root_dir='./boston_stacking')
    concat_feat = [copy_feat, process_feat]

    singles = [
        XGBoostRegressorOutOfFold(name='xgb_simple', parent=concat_feat),
        RFRegressorFeatureOutOfFold(name='rf', parent=concat_feat),
        KNeighborRegressorOutOfFold(name='kneighbor', parent=concat_feat),
        OptunaXGBRegressionOutOfFold(name='xgb_optuna', n_trials=20, parent=concat_feat),
        # seed averaging block
        create_boosting_seed_blocks(feature_class=XGBoostRegressorOutOfFold, prefix='xgb_', parent=concat_feat),
        create_boosting_seed_blocks(feature_class=LGBMRegressorOutOfFold, prefix='lgbm_', parent=concat_feat),

        # only processed feature
        create_boosting_seed_blocks(feature_class=LGBMRegressorOutOfFold, prefix='only_process_lgbm_',
                                    parent=process_feat)
    ]
    ens = EnsembleFeature(name='ensumble', parent=singles)  # ensemble of stackings

    # create stacking models
    stackings = [
        # ridge model has single models as input
        RidgeOutOfFold(name='stacking_ridge', parent=singles, n_trials=10),
        # xgboost parameter tuned by optuna
        OptunaXGBRegressionOutOfFold(name='stacking_xgb', parent=singles, n_trials=100),
    ]
    stacking_stacking_knn \
        = KNeighborRegressorOutOfFold(name='stacking_stacking_knn', parent=stackings)
    naive_xgb = XGBoostRegressorOutOfFold(name='naive_xgb', parent=copy_feat)

    ens_all = RidgeOutOfFold(name='all_ridge', parent=[*singles, *stackings, ens, stacking_stacking_knn, naive_xgb])

    ens_all.fit(train_df, y)


if __name__ == '__main__':
    main()
