# coding: utf-8
"""学習とかで使う汎用的な関数などを定義する
"""

from contextlib import contextmanager
from glob import glob
from logging import getLogger, StreamHandler, FileHandler, Formatter
from time import time

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm


def set_optuna_silent():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def set_default_style(style='ticks', font='Noto Sans CJK JP', colors=None):
    """
    matplotlib, seaborn でのグラフ描写スタイルを標準的仕様に設定するメソッド
    このメソッドの呼び出しは破壊的です。

    Args:
        style(str):
        font(str):
        colors(None | list[str]):

    Returns: None

    """
    sns.set(style=style, font=font)
    if colors is None:
        colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
        sns.set_palette(sns.xkcd_palette(colors))
    return


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]'):
    start = time()
    yield
    d = time() - start
    if logger:
        logger.info(format_str.format(d))
    else:
        print(d)


def get_logger(name, log_level="DEBUG",
               output_file=None,
               handler_level="INFO",
               output_level='DEBUG',
               format_str="[%(asctime)s] %(message)s"):
    """
    :param str name:
    :param str log_level:
    :param str | None output_file:
    :return: logger
    """
    logger = getLogger(name)

    formatter = Formatter(format_str)

    handler = StreamHandler()
    logger.setLevel(log_level)
    handler.setLevel(handler_level)
    handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(handler)

    if output_file:
        file_handler = FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(output_level)
        logger.addHandler(file_handler)

    return logger


def get_train_valid_set(fold, X, y):
    if len(X) != len(y):
        raise ValueError()

    items = []
    for train_idx, test_idx in fold.split(X, y):
        item = (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx]), (train_idx, test_idx)
        items.append(item)
    return items


def get_sample_pos_weight(y):
    """

    Args:
        y(np.ndarray): shape = (n_samples, )

    Returns:
        float
    """
    unique, count = np.unique(y, return_counts=True)
    y_sample_weight = dict(zip(unique, count))
    sample_pos_weight = y_sample_weight[0] / y_sample_weight[1]
    return sample_pos_weight


def read_multiple_csv(glob_path, max_cols=1):
    """
    複数の csv を読み込んで列方向に merge する

    Args:
        glob_path(str): 読み込むファイルへのパスの glob text.
            例えば `"./data/**/*.csv"` など
        max_cols(int): これ以上の column を持つ csv は skip する
            例えば 1 が設定されるとカラムが 2 以上のものを無視する

    Returns:

    """
    path_list = glob(glob_path)
    df = pd.DataFrame()
    for p in path_list:
        _df = pd.read_csv(p)
        if len(_df.columns) > max_cols:
            continue
        df = pd.concat([df, _df], axis=1)
    return df


logger = get_logger(__name__)


def download_from_gdrive(id, destination):
    """
    Download file from Google Drive
    :param str id: g-drive id
    :param str destination: output path
    :return:
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        logger.info("get download warning. set confirm token.")
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    """
    verify whether warned or not.

    [note] In Google Drive Api, if requests content size is large,
    the user are send to verification page.

    :param requests.Response response:
    :return:
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v

    return None


def save_response_content(response, destination):
    """
    :param requests.Response response:
    :param str destination:
    :return:
    """
    chunk_size = 1024 * 1024
    logger.info("start downloading...")
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), unit="MB"):
            f.write(chunk)
    logger.info("Finish!!")
    logger.info("Save to:{}".format(destination))
