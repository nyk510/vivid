import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from vivid.text import normalize_neologd


def check_has_column(df_input, columns):
    not_have = [n for n in columns if n not in df_input.columns]
    return not_have


class NotMatchLength(Exception):
    pass


class NotFittedError(Exception):
    pass


class AbstractAtom(TransformerMixin):
    use_columns = None

    def __str__(self):
        return self.__class__.__name__

    def __init__(self):
        self._check_implement()

    def __call__(self, df_input, y=None):
        self.generate(df_input, y)

    def _check_implement(self):
        if self.use_columns is None:
            return

        if isinstance(self.use_columns, str):
            raise TypeError('')

        try:
            cols = iter(self.use_columns)
        except TypeError as e:
            raise TypeError('Invalid type `use_columns` assigned on {}. Must iterable obj'.format(self))

        if not all([isinstance(c, str) for c in cols]):
            raise ValueError('None Sting Value contains in `use_cols`. {}'.format(self.use_columns))

    @property
    def is_fitted(self):
        """"""
        return True

    def _check_input(self, df_input, y=None):
        if self.use_columns is None:
            return

        not_have = check_has_column(df_input, self.use_columns)

        if len(not_have) > 0:
            raise ValueError('use columns `"{}"` does not exist in your input dataframe'.format(','.join(not_have)) + \
                             '\ncheck your use_columns in class:{}'.format(self.__class__.__name__))

        if y is None and not self.is_fitted:
            raise NotFittedError()

        return

    def _post_generate(self, df_input, df_out):
        if len(df_input) != len(df_out):
            raise NotMatchLength('Input length:{}, but output is: {}@{}'.format(len(df_input), len(df_out), self))
        return

    def generate(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        self._check_input(input_df, y)
        if y is None:
            out_df = self.transform(input_df)
        else:
            out_df = self.fit_transform(input_df, y)
        self._post_generate(input_df, out_df)
        return out_df

    def fit(self, input_df: pd.DataFrame, y=None):
        return self

    def transform(self, input_df) -> pd.DataFrame:
        raise NotImplementedError()


class StringContainsAtom(AbstractAtom):
    """
    文字列が指定されたカラムに含まれているかどうかを特徴とする Atom
    queryset の key に目的のカラム, 値に調べたい文字列の配列を指定します.

    Notes:
        対象となるのは string が格納されているカラムであることに注意してください.
        カラムを文字列化する処理は行わないため数値化ラムを key に指定するとエラーになります.
        同様に value として与える list に対しても string の処理は行っていないためこちらもエラーになります

    Examples:
        class TitleContainsAtom(StringContainsAtom):
            queryset = {
                'title': ['a', 'b'],
                'place': ['osaka', 'o']
            }

    """
    queryset = {}
    preprocess = 'default_loader'

    @property
    def use_columns(self):
        return list(self.queryset.keys())

    def run_preprocess(self, s):
        if self.preprocess is None:
            return s

        if callable(self.preprocess):
            return self.preprocess(s)

        if self.preprocess != 'default_loader':
            raise ValueError('preprocess must be callable obj. {}'.format(self))

        try:
            s = s.lower()
            s = normalize_neologd(s)
            return s
        except:
            return ''

    def transform(self, input_df):
        df_out = pd.DataFrame()

        for key, queries in self.queryset.items():

            master_docs = [self.run_preprocess(d) for d in input_df[key].values]

            for q in queries:
                new_colname = '{}_{}'.format(key, q)
                df_out[new_colname] = np.where([q in doc for doc in master_docs], 1, 0)
        return df_out


class AbstractMergeAtom(AbstractAtom):
    """
    入力されたデータと外部のデータとを結合する特徴です
    実装する際には以下の3つの attribute, メソッド を実装してください

    * merge_key: どのカラムで入力データとマージを行うか
    * read_outer_dataframe: 結合する外部データを読み込むためのメソッド
    * generate_master_feature:
        外部データを加工して merge_key + マージしたい特徴カラム を作成するためのメソッド
    """

    merge_key = None
    _master_dataframe = None

    def _check_implement(self):
        if self.merge_key is None:
            raise AttributeError('{} must define merge key'.format(self.__class__.__name__))

        if not isinstance(self.merge_key, str):
            raise ValueError('merge key must set string. actually: {}'.format(self.merge_key))

        super(AbstractMergeAtom, self)._check_implement()

    @property
    def use_columns(self):
        return [self.merge_key]

    def read_outer_dataframe(self):
        """

        Returns(pd.DataFrame):

        """
        raise NotImplementedError()

    @property
    def df_outer(self):
        if self._master_dataframe is None:
            self._master_dataframe = self.read_outer_dataframe()
        return self._master_dataframe

    @df_outer.setter
    def df_outer(self, v):
        raise ValueError('outer data does not set')

    def generate_outer_feature(self):
        """

        Returns:
            pd.DataFrame
                マスターデータから作成した特徴量
        """
        raise NotImplementedError()

    def transform(self, input_df):
        df_outer_feature = self.generate_outer_feature()

        if self.merge_key not in df_outer_feature.columns:
            df_outer_feature[self.merge_key] = self.df_outer[self.merge_key].copy()

        df_out = pd.merge(input_df[[self.merge_key]], df_outer_feature, on=self.merge_key, how='left')
        df_out = df_out.drop(columns=[self.merge_key])
        return df_out
