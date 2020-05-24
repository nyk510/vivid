import pandas as pd
from sklearn.datasets import load_boston

from vivid.estimators.boosting import XGBRegressorBlock

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    xgb = XGBRegressorBlock('xgb')
    xgb.fit(train_df, y)

    xgb.predict(train_df)
