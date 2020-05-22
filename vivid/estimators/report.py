from typing import Union, Callable

import matplotlib.pyplot as plt
from tabulate import tabulate

from vivid.core import AbstractEvaluation, EvaluationEnv
from vivid.metrics import regression_metrics, binary_metrics
from vivid.estimators.base import MetaBlock
from vivid.visualize import NotSupportedError, visualize_feature_importance, visualize_distributions, \
    visualize_pr_curve, \
    visualize_roc_auc_curve


class MetricReport(AbstractEvaluation):
    """calculate predict score and logging

    Notes:
        when use this class, `feature_instance` expect `is_regression_model` attribute.
        If instance does not have, skip calculation
    """

    def __init__(self, show_to_log=True):
        self.show_to_log = show_to_log

    def call(self, env: EvaluationEnv):
        if not env.block.is_estimator:
            return

        if not isinstance(env.block, MetaBlock):
            return

        y = env.y
        oof = env.output_df.values[:, 0]
        experiment = env.experiment
        if env.block.is_regression_model:
            metric_df = regression_metrics(y, oof)
        else:
            metric_df = binary_metrics(y, oof)

        experiment.mark('train_metrics', metric_df['score'])

        if not self.show_to_log:
            return
        s_metric = tabulate(metric_df.T, headers='keys')
        experiment.logger.info(s_metric)


class FeatureImportanceReport(AbstractEvaluation):
    """plot feature importance report"""

    def __init__(self, n_importance_plot=50):
        super(FeatureImportanceReport, self).__init__()
        self.n_importance_plot = n_importance_plot

    def call(self, env: EvaluationEnv):
        if not isinstance(env.block, MetaBlock):
            return

        try:
            fig, ax, importance_df = visualize_feature_importance(env.block._fitted_models,
                                                                  columns=env.parent_df.columns,
                                                                  top_n=self.n_importance_plot,
                                                                  plot_type='boxen')

        except NotSupportedError:
            env.experiment.logger.warning(f'class {env.block.model_class} is not supported for feature importance.')
            return

        env.experiment.save_figure('importance', fig)
        plt.close(fig)

        env.experiment.save_dataframe('importance', importance_df)


class CurveFigureReport(AbstractEvaluation):
    def __init__(self, visualizer: Union[None, Callable] = visualize_distributions, name=None):
        super(CurveFigureReport, self).__init__()
        self.visualizer = visualizer
        self.name = name

    def call(self, env: EvaluationEnv):
        fig, ax = self.visualizer(y_true=env.y, y_pred=env.output_df.values[:, 0], )
        env.experiment.save_figure(self.name, fig=fig)


def curve_figure_block():
    visualizers = [
        visualize_roc_auc_curve,
        visualize_distributions,
        visualize_pr_curve
    ]

    return [CurveFigureReport(v, name=v.__name__.replace('visualize_', '')) for v in visualizers]
