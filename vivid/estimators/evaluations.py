from typing import Callable, List

from tabulate import tabulate

from vivid.core import AbstractEvaluation, EvaluationEnv
from vivid.metrics import regression_metrics, binary_metrics
from vivid.visualize import NotSupportedError, visualize_feature_importance, visualize_distributions, \
    visualize_pr_curve, visualize_roc_auc_curve


def estimator_only(func):
    def wrapper(self, env: EvaluationEnv):
        if not env.block.is_estimator:
            return

        return func(self, env)

    return wrapper


class MetricReport(AbstractEvaluation):
    """calculate predict score and logging

    Notes:
        when use this class, `feature_instance` expect `is_regression_model` attribute.
        If instance does not have, skip calculation
    """

    def __init__(self, show_to_log=True):
        self.show_to_log = show_to_log

    @estimator_only
    def call(self, env: EvaluationEnv):

        if not hasattr(env.block, 'is_regression_model'):
            return

        y = env.y
        oof = env.output_df.values[:, 0]
        experiment = env.experiment

        if env.block.is_regression_model:
            score = regression_metrics(y, oof)
        else:
            score = binary_metrics(y, oof)
        experiment.mark('train_metrics', score)

        if not self.show_to_log:
            return
        s_metric = tabulate([score], headers='keys', tablefmt='github')
        for l in s_metric.split('\n'):
            experiment.logger.info(l)


class FeatureImportanceReport(AbstractEvaluation):
    """plot feature importance report"""

    def __init__(self, n_importance_plot=50):
        super(FeatureImportanceReport, self).__init__()
        self.n_importance_plot = n_importance_plot

    @estimator_only
    def call(self, env: EvaluationEnv):
        try:
            env.experiment.logger.debug('start plot importance')
            fig, ax, importance_df = visualize_feature_importance(env.block._fitted_models,
                                                                  columns=env.parent_df.columns,
                                                                  top_n=self.n_importance_plot,
                                                                  plot_type='boxen')
        except NotSupportedError:
            env.experiment.logger.debug(f'class {env.block.model_class} is not supported for feature importance.')
            return

        env.experiment.save_figure('importance', fig)
        env.experiment.save_dataframe('importance', importance_df)


class CurveFigureReport(AbstractEvaluation):
    """
    正解ラベル y_true と予測値 y_pred を引数に持ち, fig, ax を返す関数を使った可視化結果を保存する report
    """

    def __init__(self, visualizer: Callable, name=None):
        super(CurveFigureReport, self).__init__()
        self.visualizer = visualizer
        self.name = name

    @estimator_only
    def call(self, env: EvaluationEnv):
        if env.block.is_regression_model:
            return
        fig, ax = self.visualizer(y_true=env.y, y_pred=env.output_df.values[:, 0], )
        env.experiment.save_figure(self.name, fig=fig)


def curve_figure_reports() -> List[AbstractEvaluation]:
    visualizers = [
        visualize_roc_auc_curve,
        visualize_distributions,
        visualize_pr_curve
    ]

    return [CurveFigureReport(v, name=v.__name__.replace('visualize_', '')) for v in visualizers]
