import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import AbstractFeature
from vivid.metrics import regression_metrics
from vivid.out_of_fold.boosting import XGBoostRegressorOutOfFold


class BostonProcessFeature(AbstractFeature):
    def call(self, df_source: pd.DataFrame, y=None, test=False):
        return df_source


def main():
    X, y = load_boston(return_X_y=True)
    df_x = pd.DataFrame(X)

    entry = BostonProcessFeature(name='boston_base', root_dir='./boston_simple')  # output to `./boston_simple`

    basic_xgb_feature = XGBoostRegressorOutOfFold(name='xgb_simple', parent=entry)  # normal XGBoost Model
    df = basic_xgb_feature.fit(df_x, y, force=True)  # fit

    for i, cols in df.T.iterrows():
        score = regression_metrics(y, cols.values)  # calculate regression metrics
        print(cols.name, score)


if __name__ == '__main__':
    main()
