# coding: utf-8
"""特徴量や予測値の可視化を手助けする tool の定義
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import cm
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .sklearn_extend import PrePostProcessModel
from .utils import get_logger

logger = get_logger(__name__)


def xgb_feature_importance(x, y, columns):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=71)

    unique, count = np.unique(y_train, return_counts=True)
    y_sample_weight = dict(zip(unique, count))
    sample_pos_weight = y_sample_weight[0] / y_sample_weight[1]
    print('pos/neg samples:', y_sample_weight)

    xgb_model = xgb.XGBClassifier(scale_pos_weight=sample_pos_weight,
                                  gamma=1e-4,
                                  reg_lambda=1e-1,
                                  learning_rate=.1,
                                  n_estimators=400,
                                  colsample_bytree=.8)

    xgb_model.fit(x_train, y_train, verbose=True
                  , early_stopping_rounds=20, eval_metric="auc",
                  eval_set=[(x_test, y_test)])

    def calc_test_metrics(y_true, x_test, model):
        auc = roc_auc_score(y_true, model.predict_proba(x_test)[:, 1])
        acc = accuracy_score(y_true, model.predict(x_test))
        return auc, acc

    auc, acc = calc_test_metrics(y_test, x_test, xgb_model)
    print('Validation AUC: {auc:.5f} acc: {acc:.5f}'.format(**locals()))

    df_feature_importance = pd.DataFrame(data=[xgb_model.feature_importances_], columns=columns,
                                         index=['feature_importance'])
    df_feature_importance = df_feature_importance.T.sort_values('feature_importance')
    axis = df_feature_importance.plot(kind='barh', figsize=(6, len(df_feature_importance.index) * .3))

    return axis


def corr_euclid_clustermap(df, n_rows=None, n_cols=None, cmap='viridis', z_score=None, **kwargs):
    """
    seaborn.clustermap を row に対して euclid を, column に対して correlation で実行する
    Args:
        df(pd.DataFrame):
        n_rows(int | float | None):
        n_cols(int | float | None):
        cmap(str):
        kwargs:

    Returns:
        sns.Grid

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


def plot_value_count_summary(df: pd.DataFrame, column_name, color='C0', normalize=False):
    col = df[column_name]
    vc_i = col.value_counts(dropna=False, normalize=normalize)
    count_null = col.isnull().sum()

    fig = plt.figure(figsize=(6, 4))
    ax_i = fig.add_subplot(1, 1, 1)
    ax_i = vc_i.plot(kind='barh', ax=ax_i, color=color)
    ax_i.set_title('{column_name} null count: {count_null}'.format(**locals()))
    ax_i.set_xlabel('#Count/#All (ratio)')
    return fig, ax_i


def plot_predict_distribution(df_pred, output_dir):
    """
    予測モデルの予測値分布が target の値によってどのように違うのかを可視化する plot
    Args:
        df_pred(pd.DataFrame): 予測値をカラムに持つデータフレーム.
            `"target"` カラムに正解ラベルを格納して渡す.
            target は {0, 1} を値に持つ必要がある.
        output_dir(str): 画像を保存するディレクトリへのパス

    Returns:

    """
    for model_name in df_pred.columns:
        if model_name == 'target':
            continue
        g = sns.catplot(data=df_pred, x='target', y=model_name, kind='swarm')
        g.fig.suptitle('Predict Distribution')
        g.savefig(os.path.join(output_dir, 'target_swarm_y={}.png'.format(model_name)), dpi=150)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        s = df_pred[model_name]
        s_pos = s[df_pred.target == 1]
        s_neg = s[df_pred.target == 0]

        try:
            sns.distplot(s_neg, rug=True, hist=False, label='Target = 0 (Negative)', ax=ax)
            sns.distplot(s_pos, rug=True, hist=False, label='Target = 1 (Positive)', ax=ax)
        except np.linalg.LinAlgError as e:
            logger.warning('distribution estimation is failed with {}'.format(e))

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'target_dist_y={}.png'.format(model_name)), dpi=150)
        plt.close('all')


def calculate_scores(df_pred, y_true):
    """
    二値分類に関する score を網羅的に計算した DataFrame を計算する

    Args:
        df_pred(pd.DataFrame): 各カラムに予測値の入ったデータフレーム. shape = (n_samples, n_estimators,)
        y_true(np.ndarray): shape = (n_samples,)

    Returns:
        pd.DataFrame
    """
    scores = []
    cols = []
    for name, col in df_pred.T.iterrows():
        if name == 'target':
            continue
        auc_i = roc_auc_score(y_true, col)
        logloss_i = log_loss(y_true, col)

        # Note: しきい値は 0.5 で決め打ちになっている
        pred_label = list(np.where(col >= .5, 1, 0))
        acc = accuracy_score(y_true, pred_label)
        recall_i = recall_score(y_true, pred_label)

        # f1/precision は予測ラベルが positive の集合が分母になるためそもそも positive の予測がないときには計算不能
        # 無理やり計算しようとするとエラーになるため初めに予測値のうち positive のものの数を計算し場合分けする 
        n_pred_pos = sum(pred_label)
        if n_pred_pos < 1:
            f1 = precision_i = 0
        else:
            f1 = f1_score(y_true, pred_label, )
            precision_i = precision_score(y_true, pred_label)

        scores.append([auc_i, logloss_i, f1, acc, recall_i, precision_i])
        cols.append(name)

    df_score = pd.DataFrame(scores, index=cols,
                            columns=['auc_score', 'negative_logloss', 'f1_score', 'accuracy', 'recall', 'precision'])
    df_score = df_score.sort_values('auc_score')
    return df_score


def plot_auc_curve(df_pred, y_true):
    """

    Args:
        df_pred(pd.DataFrame):
        y_true(np.ndarray):

    Returns:

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    _df = df_pred.copy()
    if 'target' in _df.columns:
        del _df['target']

    # AUC の高い順に plot したいのでその順序 `ordering` を始めに算出する
    # TODO: すべてのメトリックを計算することになっているので無駄が多い. metric 指定で計算できるようにしたい.
    df_score = calculate_scores(_df, y_true)
    ordering = df_score.sort_values('auc_score', ascending=False).index
    for i, (name, pred) in enumerate(_df[ordering].T.iterrows()):
        if name == 'target':
            continue

        fpr, pfr, _ = roc_curve(y_true, pred)
        ax.plot(fpr, pfr, label=name, color=cm.viridis(i / len(df_pred.columns)))
    # 対角線に基準線 (AUC=0.5) を引く
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='grey')
    fig.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax


def visualize_feature_importance(models, columns, plot_type='bar', ax=None, top_n=None, **plot_kwgs):
    """
    学習済みの Boosting Model から feature importance を plot

    Args:
        models: 学習済み Boosting モデル (LightGBM or XGBoost)
        columns: 特徴量の名前の List
        plot_type: `"bar"` or `"boxen"`
        top_n: int が指定された時, 上位 n 件を plot します
        ax: matplotlib.ax object. None のとき新しく fig, ax を作成します
        **plot_kwgs:

    Returns:
        ax is None, return fig, ax, feature importance df
        else: return ax, feature importance df
    """

    # set matplotlib loglevel 'ERROR' (avoid 'c' argument looks like a single numeric RGB or RGBA sequence)
    # たぶん seaborn のバグ?
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    importance_df = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        if isinstance(model, PrePostProcessModel):
            _df['feature_importance'] = model.fitted_model_.feature_importances_
        else:
            _df['feature_importance'] = model.feature_importances_
        _df['column'] = columns
        _df['fold'] = i + 1
        importance_df = pd.concat([importance_df, _df], axis=0, ignore_index=True)

    order = importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',
                                                                                      ascending=False).index

    if isinstance(top_n, int):
        order = order[:top_n]

    if ax is None:
        fig = plt.figure(figsize=(7, len(columns) * .3))
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
        'palette': 'cividis'
    }
    params.update(plot_kwgs)

    if plot_type == 'boxen':
        sns.boxenplot(**params)
    elif plot_type == 'bar':
        sns.barplot(**params)
    else:
        raise ValueError('plot_type must be in boxen or bar. Actually, {}'.format(plot_type))
    ax.tick_params(axis='x', rotation=90)

    if fig is None:
        return ax, importance_df
    fig.tight_layout()
    return fig, ax, importance_df
