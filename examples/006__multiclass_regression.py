import pandas as pd
from sklearn.datasets import load_iris

from vivid.backends.experiments import LocalExperimentBackend
from vivid.estimators.boosting import XGBClassifierBlock

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    train_df = pd.DataFrame(X)
    experiment = LocalExperimentBackend('./outputs/006__multiclass')

    xgb = XGBClassifierBlock('xgb_classifier', add_init_param={'eval_metric': 'mlogloss'})

    # use runner (recommended)
    from vivid.runner import create_runner

    runner = create_runner(blocks=xgb, experiment=experiment)
    runner.fit(train_df, y, ignore_past_log=True)
    pred = runner.predict(train_df)

    print(pred[0].out_df)