import numpy as np
import pandas as pd
import pytest

from vivid.featureset.encodings import CountEncodingAtom, OneHotEncodingAtom, InnerMergeAtom


class BaseTestCase:
    def setup_method(self):
        data = [
            [1, 2.1, 'hoge'],
            [1, 1.01, 'spam'],
            [2, 10.001, 'ham'],
            [1, 1.1, 'spam'],
            [3, 2.5, 'spam'],
            [1, None, None]
        ]
        self.train_df = pd.DataFrame(data, columns=['int1', 'float1', 'str1'])
        self.y = [1] * len(self.train_df)

        test_data = [
            data[0], data[2]
        ]
        self.test_df = pd.DataFrame(test_data, columns=self.train_df.columns)

    def is_generate_idempotency(self, atom):
        """atom のべき等チェック"""
        feat_1 = atom.generate(self.train_df, self.y)
        feat_2 = atom.generate(self.train_df)

        return feat_1.equals(feat_2)


class TestCountEncodingAtom(BaseTestCase):
    def setup_method(self):
        super(TestCountEncodingAtom, self).setup_method()

        class IrisCountEncodingAtom(CountEncodingAtom):
            use_columns = ['int1', 'str1']

        self.atom = IrisCountEncodingAtom()

    def test_generate_data(self):
        feat_train = self.atom.generate(self.train_df, self.y)

        assert len(self.train_df) == len(feat_train)

    def test_output_values(self):
        """出力データが正しいことの確認"""

        # 学習データで学習済み
        self.atom.generate(self.train_df, self.y)

        test_data = [
            [1, 'spam'],  # 対応関係があるもの
            [2, 'ham'],
            [None, None]  # レコードにない or None
        ]

        ground_truth = [
            [4, 3],
            [1, 1],
            [np.nan, np.nan]
        ]
        test_df = pd.DataFrame(test_data, columns=self.atom.use_columns)

        feat_test = self.atom.generate(test_df)
        assert len(test_df) == len(feat_test)
        assert pd.DataFrame(ground_truth).equals(pd.DataFrame(feat_test.values))

    def test_generate_idempotency(self):
        assert self.is_generate_idempotency(self.atom)

    def test_null_contains(self):
        feat_train = self.atom.generate(self.train_df, self.y)


class TestOneHotEncodingAtom(BaseTestCase):
    def setup_method(self):
        super(TestOneHotEncodingAtom, self).setup_method()

        class IrisOneHotAtom(OneHotEncodingAtom):
            use_columns = ['int1']

        self.atom = IrisOneHotAtom()

    def test_generate(self):
        self.atom.generate(self.train_df, self.y)
        self.atom.generate(self.test_df)

    def test_generate_idempotency(self):
        assert self.is_generate_idempotency(self.atom)

    def test_null_contains(self):
        class Atom(OneHotEncodingAtom):
            use_columns = ['str1']

        atom = Atom()
        feat_train = atom.generate(self.train_df, self.y)


class TestInnerMergeAtom(BaseTestCase):
    def setup_method(self):
        super(TestInnerMergeAtom, self).setup_method()

        class IrisInnerMergeAtom(InnerMergeAtom):
            use_columns = ['int1', 'float1']

        self.atom_class = IrisInnerMergeAtom

    @pytest.mark.parametrize('agg', [
        'mean', 'std', 'median', 'min', 'max'
    ])
    def test_generate(self, agg):
        atom = self.atom_class(merge_key='int1', agg=agg)

        atom.generate(self.train_df, self.y)
        atom.generate(self.test_df)

        assert self.is_generate_idempotency(atom)
