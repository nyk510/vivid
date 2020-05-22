import pandas as pd
from sklearn.datasets import load_boston

from vivid.model_selection import StratifiedGroupKFold
from vivid.estimators.boosting import XGBRegressorBlock

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    cv = StratifiedGroupKFold(n_splits=5, q=10)
    xgb = XGBRegressorBlock('xgb', cv=cv)

    xgb.fit(train_df, y)
