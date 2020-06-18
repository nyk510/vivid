import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from vivid.backends.experiments import LocalExperimentBackend
from vivid.estimators.boosting import XGBRegressorBlock, XGBClassifierBlock

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    y_classifier = np.where(y > 10., 1, 0)
    train_df = pd.DataFrame(X)
    experiment = LocalExperimentBackend('./outputs/simple')

    xgb = XGBRegressorBlock('xgb_reg')
    with experiment.as_environment(xgb.runtime_env) as exp:
        oof = xgb.fit(train_df, y, experiment=exp)
        xgb.report(train_df, oof, y, experiment=exp)

    xgb = XGBClassifierBlock('xgb_classifier')
    with experiment.as_environment(xgb.runtime_env) as exp:
        oof = xgb.fit(train_df, y_classifier, experiment=exp)
        xgb.report(train_df, oof, y_classifier, experiment=exp)

    # use runner (recommended)
    from vivid.runner import create_runner

    runner = create_runner(blocks=xgb, experiment=experiment)
    runner.fit(train_df, y_classifier, ignore_past_log=True)
    runner.predict(train_df)
