# coding: utf-8
"""
"""

import os

import pandas as pd

from .env import Settings
from .utils import get_logger, timer


class AbstractFeature(object):
    """
    特徴量の抽象クラス. すべての特徴量はこのクラスを継承して作成する.
    """

    # True のとき local のディレクトリ内に作成したモデル及び特徴量を保存する
    allow_save_local = True

    def __init__(self,
                 name,
                 parent=None,
                 root_dir=None):
        """

        Args:
            name(str):
            parent(AbstractFeature | None):
            root_dir(str | None): 特徴量を保存するディレクトリへのパス.
                指定されないとき `parent_feature` の `root_dir` をチェックし
        """
        self.parent = parent

        # 親ディレクトリがない時それは即ち entry point (root node)
        self.is_entrypoint = parent is None
        self.name = name
        self._root_dir = root_dir

        # 学習時に作成した特徴量
        self.feat_on_train_df = None
        self.feat_on_test_df = None
        self.initialize()

    def __str__(self):
        if self.parent is None:
            return self.name
        return '{}__{}'.format(str(self.name), str(self.parent))

    @property
    def root_dir(self):
        """

        Returns:
            str:
        """
        # この特徴量にディレクトリが指定されているときそれを返す
        if self._root_dir is not None:
            return self._root_dir

        # ディレクトリ指定がなく, もと特徴量も無いとき `None` を返す
        if self.parent is None:
            return None

        # ディレクトリ指定は無いが元特徴量があるとき, そちらに委ねる
        return self.parent.root_dir

    @property
    def has_output_dir(self):
        return self.root_dir is not None

    @property
    def is_recording(self):
        return self.allow_save_local and self.has_output_dir

    @property
    def output_dir(self):
        """
        モデルや特徴量を保存するディレクトリへの path
        Returns:
            str: path string

        """
        if not self.has_output_dir:
            return None
        return os.path.join(self.root_dir, str(self))

    @property
    def output_train_meta_path(self):
        """
        作成した training 時の特徴量 csv への path
        Returns:
            str: path to training csv created by myself

        """
        if not self.has_output_dir:
            return None
        return os.path.join(self.output_dir, 'train.csv')

    @property
    def has_train_meta_path(self):
        return self.output_train_meta_path is not None and \
               os.path.exists(self.output_train_meta_path)

    @property
    def has_parent(self):
        return self.parent is not None

    def call(self, df_source: pd.DataFrame, y=None, test=False) -> pd.DataFrame:
        """

        Args:
            df_source(pd.DataFrame):
            y(np.array): shape = (n_samples, )
            test(boolean):

        Returns:
            pd.DataFrame
        """
        raise NotImplementedError()

    def initialize(self):
        log_output = None
        if self.is_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            log_output = os.path.join(self.output_dir, 'log.txt')

        if self.parent is None:
            logger_name = 'feature.' + self.name
        else:
            logger_name = self.parent.name + '.' + self.name

        self.logger = get_logger(f'vivid.{logger_name}',
                                 log_level=Settings.LOG_LEVEL,
                                 output_file=log_output,
                                 output_level=Settings.TXT_LOG_LEVEL,
                                 format_str='[%(asctime)s vivid.{}] %(message)s'.format(logger_name))

    def fit(self, input_df, y, force=False):
        """
        訓練データのセット (特徴のデータフレームとターゲット変数) を用いてモデルを学習し
        特徴量に変換されたデータを返す method

        Args:
            input_df(pd.DataFrame):
            y(np.array):

        Returns:
            pd.DataFrame

        """
        self.initialize()
        if self.has_train_meta_path and not force:
            self.logger.info('train data is exists. load from local.')
            return pd.read_csv(self.output_train_meta_path)

        # 学習語の特徴量がキャッシュされている時それを返す
        if self.feat_on_train_df is not None:
            return self.feat_on_train_df

        if self.has_parent:
            df_feat = self.parent.fit(input_df, y, force=force)
        else:
            df_feat = input_df

        with timer(self.logger, format_str='for create feature: {:.3f}[s]'):
            df_feat = self.call(df_feat, y)

        if Settings.CACHE_ON_TRAIN:
            self.feat_on_train_df = df_feat
        if self.is_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            assert os.path.exists(self.output_dir)
            self.logger.info('training data save to: {}'.format(self.output_train_meta_path))
            df_feat.to_csv(self.output_train_meta_path, index=False)

        self.logger.info('Shape: {}'.format(df_feat.shape))

        self.logger.debug('Column Names')
        for c in df_feat.columns:
            self.logger.debug(c)

        return df_feat

    def predict(self, input_df, force=False):
        """
        未知のデータを特徴量に変換する method

        Args:
            input_df(pd.DataFrame):
                予測したいデータの dataframe

        Returns:
            pd.DataFrame: 特徴量として変換された DataFrame
        """
        self.initialize()

        # 既に作成した特徴がある場合それを返す
        if not force and self.feat_on_test_df is not None:
            return self.feat_on_test_df
        if self.has_parent:
            df = self.parent.predict(input_df, force=force)
        else:
            df = input_df
        df_pred = self.call(df, test=True)

        if self.is_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            df_pred.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        self.feat_on_test_df = df_pred
        return df_pred


class MergeFeature(AbstractFeature):
    """
    特徴量同士を結合した特徴
    """
    # merge するだけなので保存しない
    allow_save_local = False

    def __init__(self, input_features, name=None, root_dir=None):
        """

        Args:
            input_features(list[AbstractFeature]): もととなる特徴量の list
        """
        if name is None:
            name = '-'.join([f.name for f in input_features])
        self.features = input_features

        # TODO: parent が複数あるばあいにもグラフ構造を担保して保存できるようにしたい
        super(MergeFeature, self).__init__(name, parent=None, root_dir=root_dir)

        names = [f.name for f in input_features]
        df_merge_from = pd.DataFrame(names, columns=['feature_name'])

        if self.is_recording:
            df_merge_from.to_csv(os.path.join(self.output_dir, 'source_features.csv'), index=False)

    def call(self, source_df, y=None, test=False):
        df_concat = None

        for feat in self.features:
            if test:
                _df = feat.predict(source_df)
            else:
                _df = feat.fit(source_df, y)

            if df_concat is None:
                df_concat = _df
            else:
                df_concat = pd.concat([df_concat, _df], axis=1)
        return df_concat


class EnsembleFeature(MergeFeature):
    """
    子特徴量を平均化したものを返す特徴量
    """
    allow_save_local = True

    def call(self, df_source, y=None, test=False):
        df = super(EnsembleFeature, self).call(df_source, y, test)
        return pd.DataFrame(df.mean(axis=1), columns=[self.name])
