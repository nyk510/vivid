import pandas as pd
from sklearn.datasets import load_boston

from vivid.backends.experiments import LocalExperimentBackend
from vivid.core import BaseBlock
from vivid.estimators.base import EnsembleBlock
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.estimators.ensumble import RFRegressorBlock
from vivid.estimators.linear import TunedRidgeBlock
from vivid.estimators.neural_network import KerasRegressorBlock
from vivid.estimators.svm import SVRBlock
from vivid.features.blocks import BinningCountBlock, FilterBlock


class SumBlock(BaseBlock):
    def transform(self, source_df: pd.DataFrame):
        x = source_df.sum(axis=1)
        out_df = pd.DataFrame({'sum': x})
        return out_df


if __name__ == '__main__':
    features = [
        FilterBlock(name='copy', column='__all__'),
        BinningCountBlock('bin', column='__all__'),
        SumBlock(name='sum')
    ]

    models = [
        XGBRegressorBlock(name='xgb', parent=features),
        TunedRidgeBlock('ridge', parent=features),
        RFRegressorBlock('rf', parent=features),
        KerasRegressorBlock('keras', parent=features),
        SVRBlock('svr', parent=features)
    ]

    ensemble = EnsembleBlock('ens', parent=models)
    stacking = [
        SVRBlock('svr', parent=models),
        TunedRidgeBlock('ridge', parent=models)
    ]
    stacked_stack = TunedRidgeBlock('ridge', parent=[*stacking, *models, *features])

    print(stacked_stack.show_network())
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    experiment = LocalExperimentBackend(namespace='./outputs/ensemble')

    stacked_stack.fit(train_df, y, experiment=experiment)
    stacked_stack.predict(train_df, experiment=experiment, ignore_storage=True)

    other_model = XGBRegressorBlock('xgb2', parent=features)
    other_model.fit(train_df, y, experiment=experiment)

    # ensemble model already fitted,
    ensemble.fit(train_df, y, experiment=experiment)

    estimators = set([b for b in stacked_stack.all_network_blocks() if b.is_estimator])
    for b in estimators:
        print('predict {}'.format(b.name))
        b.predict(train_df)
