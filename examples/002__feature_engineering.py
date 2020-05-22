import pandas as pd
from sklearn.datasets import load_boston

from vivid.core import BaseBlock
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.features.blocks import BinningCountBlock


class SumBlock(BaseBlock):
    def transform(self, source_df: pd.DataFrame):
        x = source_df.sum(axis=1)
        out_df = pd.DataFrame({'sum': x})
        return out_df


class CopyBlock(BaseBlock):
    def transform(self, source_df):
        return source_df


if __name__ == '__main__':
    features = [
        CopyBlock(name='copy'),
        BinningCountBlock('bin', column=0),
        SumBlock(name='sum')
    ]

    xgb = XGBRegressorBlock(name='xgb', parent=features)

    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)
    xgb.fit(train_df, y)
    xgb.predict(train_df)
