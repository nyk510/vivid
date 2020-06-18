import pandas as pd
from sklearn.datasets import load_boston

from vivid.backends.experiments import LocalExperimentBackend
from vivid.estimators.boosting import XGBRegressorBlock

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    xgb = XGBRegressorBlock('xgb')

    experiment = LocalExperimentBackend('./outputs/simple')

    with experiment.as_environment(xgb.runtime_env) as exp:
        oof = xgb.fit(train_df, y, experiment=exp)
        xgb.report(train_df, oof, y, experiment=exp)
