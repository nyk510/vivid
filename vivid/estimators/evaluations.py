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

    Examples:
        simple/xgb__28163251          : INFO     |    rmsle |    rmse |   mean-ae |   median-ae |       r2 |   explained_variance |
        simple/xgb__28163251          : INFO     |----------|---------|-----------|-------------|----------|----------------------|
        simple/xgb__28163251          : INFO     | 0.178404 | 4.23411 |    2.7862 |     2.03895 | 0.787636 |             0.789955 |

    """

    def __init__(self, show_to_log=True):
        """

        Args:
            show_to_log:
                True の時 logger info に結果を pretty に print します.
        """
        self.show_to_log = show_to_log
        super(MetricReport, self).__init__()

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
        """
        Args:
            n_importance_plot:
                表示する最大のカラム数.
                多量の特徴がある場合すべてを plot すると固まってしまう (場合によってはクラッシュする) ために設定されています
        """
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

    def __init__(self, visualizer: Callable, name=None, for_classifier=False):
        """
        Args:
            visualizer:
                可視化を実行する関数.
                引数として y_true, y_pred / 返り値として [plt.Figure, plt.Axes] を返すような関数である必要があります.
                vivid.visualizer.visualize_roc_auc_curve などの関数形式を参考にしてください.
            name:
                結果を保存するときの key.
            for_classifier:
                True の時分類問題用の可視化であるとみなし, block のモデルが classifier (not regressor) ではない時に
                実行をしません.
        """
        super(CurveFigureReport, self).__init__()
        self.visualizer = visualizer
        self.name = name
        self.for_classifier = for_classifier

    @estimator_only
    def call(self, env: EvaluationEnv):
        is_regressor = env.block.is_regression_model
        if self.for_classifier and is_regressor:
            return

        env.experiment.logger.debug('start plot {}'.format(self.name))
        fig, ax = self.visualizer(y_true=env.y, y_pred=env.output_df.values[:, 0], )
        ax.set_title(env.block.name + ' ' + self.name)
        env.experiment.logger.debug('finished.')
        env.experiment.save_figure(self.name, fig=fig)


def curve_figure_reports() -> List[AbstractEvaluation]:
    return [
        CurveFigureReport(visualize_roc_auc_curve, name='roc_auc', for_classifier=True),
        CurveFigureReport(visualize_distributions, name='distribution', for_classifier=False),
        CurveFigureReport(visualize_pr_curve, name='pr_auc', for_classifier=True)
    ]
