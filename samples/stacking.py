import pandas as pd
from sklearn.datasets import load_boston
import os
from vivid.core import AbstractFeature, EnsembleFeature, MergeFeature
from vivid.featureset import AbstractAtom
from vivid.featureset.encodings import CountEncodingAtom
from vivid.metrics import regression_metrics
from vivid.out_of_fold.boosting import XGBoostRegressorOutOfFold, OptunaXGBRegressionOutOfFold, LGBMRegressorOutOfFold
from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold
from vivid.out_of_fold.kneighbor import KNeighborRegressorOutOfFold
from vivid.out_of_fold.linear import RidgeOutOfFold


class BostonBasicAtom(AbstractAtom):
    def call(self, df_input, y=None):
        return df_input.copy()


class BostonCountEncoding(CountEncodingAtom):
    use_columns = ['']


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    df_x = pd.DataFrame(X)

    entry = BostonProcessFeature(name='boston_base', root_dir='./boston_stacking')

    singles = [
        XGBoostRegressorOutOfFold(name='xgb_simple', parent=entry),
        RFRegressorFeatureOutOfFold(name='rf', parent=entry),
        KNeighborRegressorOutOfFold(name='kneighbor', parent=entry),
        OptunaXGBRegressionOutOfFold(name='xgb_optuna', n_trials=10, parent=entry),
        *create_boosting_seed_blocks(feature_class=XGBoostRegressorOutOfFold, prefix='xgb_',
                                     parent=entry),  # seed averaging models
        *create_boosting_seed_blocks(feature_class=LGBMRegressorOutOfFold, prefix='lgbm_', parent=entry)
        # lgbm averaging models
    ]
    mreged_feature = MergeFeature([*singles, entry], root_dir=entry.root_dir,
                                  name='signle_merge')  # 一段目のモデル + もとのデータを merge した特徴量
    stackings = [
        RidgeOutOfFold(name='stacking_ridge', parent=mreged_feature, n_trials=10),
        OptunaXGBRegressionOutOfFold(name='stacking_xgb', parent=mreged_feature, n_trials=10),
    ]
    ens = EnsembleFeature(stackings[:], name='ensumble', root_dir=entry.root_dir)  # stacking のアンサンブル
    stackings.append(ens)

    df = pd.DataFrame()
    for feat in [*stackings, *singles]:
        f_df = feat.fit(df_x, y)
        df = pd.concat([df, f_df], axis=1)

    score_df = pd.DataFrame()
    for i, cols in df.T.iterrows():
        df_i = regression_metrics(y, cols.values)
        score_df[i] = df_i['score']

    score_df = score_df.T.sort_values('rmse')
    score_df.to_csv(os.path.join(entry.root_dir, 'score.csv'))
    print(score_df)


if __name__ == '__main__':
    main()
