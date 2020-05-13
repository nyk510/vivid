import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import AbstractFeature
from vivid.out_of_fold.boosting import OptunaXGBRegressionOutOfFold


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    df_x = pd.DataFrame(X)

    entry = BostonProcessFeature(name='boston_base', root_dir='./boston_optuna')
    model = OptunaXGBRegressionOutOfFold(name='xgb_optuna', parent=entry, n_trials=300, scoring_strategy='whole',
                                         scoring='neg_mean_squared_log_error')

    model.fit(df_x, y)


if __name__ == '__main__':
    main()
