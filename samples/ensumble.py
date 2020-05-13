import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import AbstractFeature
from vivid.metrics import regression_metrics
from vivid.out_of_fold.base import EnsembleFeature
from vivid.out_of_fold.boosting import XGBoostRegressorOutOfFold, LGBMRegressorOutOfFold
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold
from vivid.out_of_fold.linear import RidgeOutOfFold


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    df_x = pd.DataFrame(X)

    entry = BostonProcessFeature(name='boston_base', root_dir='./boston_ens')

    singles = [
        RidgeOutOfFold(name='ridge', parent=entry),
        XGBoostRegressorOutOfFold(name='xgb', parent=entry),
        LGBMRegressorOutOfFold(name='lgbm', parent=entry),
        RFRegressorFeatureOutOfFold(name='rf', parent=entry)
    ]

    ens = EnsembleFeature(parent=singles, name='ensumble', agg='mean')
    ens2 = EnsembleFeature(parent=[ens, *singles], name='ens2', agg='mean')

    df = pd.DataFrame()
    f_df = ens2.fit(df_x, y)
    df = pd.concat([df, f_df], axis=1)

    for i, cols in df.T.iterrows():
        score = regression_metrics(y, cols.values)
        print(cols.name, score)

    ens.predict(df_x)


if __name__ == '__main__':
    main()
