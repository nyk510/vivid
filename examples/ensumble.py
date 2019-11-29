import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import AbstractFeature, EnsembleFeature
from vivid.metrics import regression_metrics
from vivid.out_of_fold.boosting import XGBoostRegressorOutOfFold, OptunaXGBRegressionOutOfFold


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    df_x = pd.DataFrame(X)

    entry = BostonProcessFeature(name='boston_base', root_dir='./boston_ens')

    basic_xgb_feature = XGBoostRegressorOutOfFold(name='xgb_simple', parent=entry)
    optuna_xgb = OptunaXGBRegressionOutOfFold(name='xgb_optuna', n_trials=10, parent=entry)
    ens = EnsembleFeature([optuna_xgb, basic_xgb_feature], name='ensumble', root_dir=entry.root_dir)

    df = pd.DataFrame()
    f_df = ens.fit(df_x, y)
    df = pd.concat([df, f_df], axis=1)

    for i, cols in df.T.iterrows():
        score = regression_metrics(y, cols.values)
        print(cols.name, score)


if __name__ == '__main__':
    main()
