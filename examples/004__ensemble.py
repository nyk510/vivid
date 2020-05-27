import pandas as pd
from sklearn.datasets import load_boston

from vivid import Runner
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
        FilterBlock(name='all', column='__all__'),
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

    stacked_stack.show_network()
    X, y = load_boston(return_X_y=True)
    train_df = pd.DataFrame(X)

    experiment = LocalExperimentBackend(namespace='./outputs/ensemble')

    stacked_runner = Runner(blocks=stacked_stack, experiment=experiment)
    stacked_runner.fit(train_df, y)
    stacked_runner.predict(train_df)
