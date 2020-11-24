"""visualization tools
"""
from typing import Union, List, Callable, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import check_classification_targets

from .sklearn_extend import PrePostProcessModel
from .utils import get_logger

logger = get_logger(__name__)


def corr_euclid_clustermap(df, n_rows=None, n_cols=None, cmap='viridis', z_score=None, **kwargs):
    """
    Example:
        >>> df = sns.load_data('iris')
        >>> corr_euclid_clustermap(df)
    """
    _df = pd.DataFrame()

    for c_name, col in df.T.iterrows():
        vc = col.value_counts()
        if len(vc) == 1:
            logger.warning('{c_name} is only one value. skip.'.format(**locals()))
            continue
        _df[c_name] = col

    feat_link = linkage(_df.T.values, method='average', metric='correlation')
    if n_rows is None:
        n_rows = len(df.columns) * .3 + 2
    if n_cols is None:
        n_cols = len(df) * .07 + 2

    grid = sns.clustermap(_df.T, figsize=(n_cols, n_rows), row_linkage=feat_link,
                          metric="euclidean", method='ward', z_score=z_score, cmap=cmap, **kwargs)
    return grid


def check_y_and_pred(y_true, y_pred) -> (np.ndarray, np.ndarray, list):
    y_true = np.asarray(y_true)

    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    classes = le.classes_
    if len(classes) == 2:
        classes = classes[:-1]
    n_classes = len(classes)

    ohe = OneHotEncoder()
    y_true = ohe.fit_transform(y_true.reshape(-1, 1)).toarray()
    y_true = y_true[:, -n_classes:]
    y_pred = np.asarray(y_pred).reshape(-1, n_classes)
    return y_true, y_pred, classes


def visualize_distributions(y_true, y_pred, ax: Union[None, plt.Axes] = None):
    try:
        check_classification_targets(y_true)
    except ValueError:
        return visualize_continuous_distributions(y_true, y_pred, ax=ax)

    y_true, y_pred, classes = check_y_and_pred(y_true, y_pred)
    n_classes = len(classes)

    if ax is None:
        fig, axes = plt.subplots(figsize=(6 * n_classes, 5), ncols=n_classes)
        if n_classes == 1:
            axes = [axes]
    else:
        fig, axes = None, [ax]  # type: (None, List[plt.Axes])

    for y, pred, ax, class_name in zip(y_true.T, y_pred.T, axes, classes):
        sns.distplot(pred[y == 1], ax=ax, label='Pos')
        sns.distplot(pred[y == 0], ax=ax, label='Neg')
        ax.set_xlabel(f'class = {classes}')

    return fig, ax


def visualize_continuous_distributions(y_true,
                                       y_pred,
                                       ax: Union[None, plt.Axes] = None) \
    -> [Union[plt.Figure, None], plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = None, ax

    sns.distplot(y_true, ax=ax, label='Target')
    sns.distplot(y_pred, ax=ax, label='Predict')
    ax.legend()
    return fig, ax


def visualize_roc_auc_curve(y_true, y_pred,
                            ax: Union[None, plt.Axes] = None,
                            label_prefix: Union[None, str] = None) -> [Union[None, plt.Figure], plt.Axes]:
    check_classification_targets(y_true)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true, y_pred, classes = check_y_and_pred(y_true, y_pred)
    n_classes = len(classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))  # type: (plt.Figure, plt.Axes)
    else:
        fig, ax = None, ax  # type: (None, plt.Axes)

    for i in range(n_classes):
        label_i = f'label = {i} / area = {roc_auc[i]:.3f}'
        if label_prefix is not None:
            label_i = f'{label_prefix} {label_i}'
        ax.plot(fpr[i], tpr[i], label=label_i)
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='grey')
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Auc Score')
    ax.legend(loc='lower right')
    return fig, ax


def visualize_pr_curve(y_true, y_pred, ax: Union[None, plt.Axes] = None,
                       label_prefix: Union[None, str] = None) -> [Union[None, plt.Figure], plt.Axes]:
    check_classification_targets(y_true)
    precision = dict()
    recall = dict()
    pr_score = dict()
    y_true, y_pred, classes = check_y_and_pred(y_true, y_pred)
    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_score[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = None, ax

    for i in range(n_classes):
        label_i = f'label = {i} / area = {pr_score[i]:.3f}'
        if label_prefix is not None:
            label_i = f'{label_prefix} {label_i}'
        ax.plot(recall[i], precision[i], label=label_i)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Recall Curve')
    ax.legend(loc='lower left')
    return fig, ax


def visualize_confusion_matrix(y_true,
                               pred_label,
                               ax: Optional[plt.Axes] = None,
                               labels: Optional[list] = None,
                               conf_options: Optional[dict] = None,
                               plot_options: Optional[dict] = None) -> Tuple[plt.Axes, np.ndarray]:
    """
    visualize confusion matrix

    Args:
        y_true:
            True Label. shape = (n_samples, )
        pred_label:
            Prediction Label. shape = (n_samples, )
        ax:
            matplotlib.pyplot.Axes object.
        labels:
            plot labels
        conf_options:
            option kwrgs when calculate confusion matrix.
            pass to `confusion_matrix` (defined at scikit-learn)
        plot_options:
            option key-words when plot seaborn heatmap

    Returns:

    """

    _conf_options = {
        'normalize': 'true',
    }
    if conf_options is not None:
        _conf_options.update(conf_options)

    _plot_options = {
        'cmap': 'Blues',
        'annot': True
    }
    if plot_options is not None:
        _plot_options.update(plot_options)

    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            **_conf_options)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, ax=ax, **_plot_options)
    ax.set_ylabel('Label')
    ax.set_xlabel('Predict')

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params('y', labelrotation=0)
        ax.tick_params('x', labelrotation=90)

    return ax, conf


class NotSupportedError(BaseException):
    pass


def extract_importance(clf: BaseEstimator):
    attrs = [
        'feature_importances_',  # scikit-learn random forest
        'feature_importance_',  # xgboost / lightgbm
        'coef_'  # scikit-learn linear model
    ]
    for att_name in attrs:
        if hasattr(clf, att_name):
            return getattr(clf, att_name)

    raise NotSupportedError(f'Cant extract model feature importance. Check the classifier {type(clf)} support.' + \
                            f'\n({clf}')


def visualize_feature_importance(models,
                                 columns: Union[None, List[str]] = None,
                                 plot_type='bar',
                                 ax: Union[None, plt.Axes] = None,
                                 top_n: Union[None, int] = None,
                                 feature_extractor: Union[None, Callable[[BaseEstimator], np.ndarray]] = None,
                                 **plot_kwgs) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    plot feature importance from a learned Model

    Currently following model are supported.
        * scikit learn's
            * linear model
            * random forest
        * xgboost (sklearn interface)
        * lightgbm (sklearn interface)

    The `extract_importance` method is used to retrieve features from a model. See also.

    Args:
        models:
            list of trained models.
        columns:
            List of names of feature
        plot_type:
            importance plot style. if set as "bar", call seaborn.barplot and "boxend" calls seaborn.boxen plot.
            `"bar"` or `"boxen"`.
        top_n:
            When int is specified, plot the top n items
        ax:
            matplotlib plt.Axes obj. Create a new fig, ax is none.
        feature_extractor:
            It is an argument for plotting a feature for an unsupported model.
            If set, the feature-grabbing method is overridden.
            Must be a function that takes model as an argument and returns a np array.
        **plot_kwgs:
            plot extra kwrgs. pass to seaborn.boxenplot or barplot function.

    Returns:
        ax is None, return fig, ax, feature importance df
        else: return ax, feature importance df
    """

    # set matplotlib loglevel 'ERROR' (avoid 'c' argument looks like a single numeric RGB or RGBA sequence)
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')

    if feature_extractor is None: feature_extractor = extract_importance
    importance_df = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        if isinstance(model, PrePostProcessModel):
            clf = model.fitted_model_
        else:
            clf = model

        importance = feature_extractor(clf)
        _df['feature_importance'] = np.array(importance).reshape(-1)
        _df['column'] = columns if columns is not None else range(len(_df))
        _df['fold'] = i + 1
        importance_df = pd.concat([importance_df, _df], axis=0, ignore_index=True)

    order = importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',
                                                                                      ascending=False).index

    if isinstance(top_n, int):
        order = order[:top_n]

    if ax is None:
        h = max(len(order) * .2, 5)
        fig = plt.figure(figsize=(7, h))
        ax = fig.add_subplot(111)
    else:
        fig = None
    params = {
        'data': importance_df,
        'x': 'feature_importance',
        'y': 'column',
        'order': order,
        'ax': ax,
        'orient': 'h',
        'palette': 'viridis'
    }
    params.update(plot_kwgs)

    if plot_type == 'boxen':
        sns.boxenplot(**params)
    elif plot_type == 'bar':
        sns.barplot(**params)
    else:
        raise ValueError('plot_type must be in boxen or bar. Actually, {}'.format(plot_type))
    ax.tick_params(axis='x', rotation=90)

    if fig:
        fig.tight_layout()
    return fig, ax, importance_df
