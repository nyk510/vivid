import pandas as pd
from sklearn.datasets import load_boston

from vivid.backends.experiments import LocalExperimentBackend
from vivid.estimators.boosting import XGBRegressorBlock

if __name__ == '__main__':
    xgb = XGBRegressorBlock(name='xgb')

    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)
    exp = LocalExperimentBackend(namespace='./simple')
    exp.clear()
    xgb.fit(train_df, y, experiment=exp)

    # re-train is no time (use cache)
    xgb.fit(train_df, y, experiment=exp)

    # switch to other environment, no longer use cache
    with exp.as_environment('new_env', style='nested') as new_exp:
        xgb.fit(train_df, y, experiment=new_exp)
