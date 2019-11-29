import pytest
import seaborn as sns

from vivid.featureset.encodings import CountEncodingAtom, OneHotEncodingAtom, InnerMergeAtom


class BaseTestCase:
    def setup_method(self):
        iris = sns.load_dataset('iris')
        idx = iris.index < 100
        self.train_df = iris[idx].reset_index(drop=True)
        self.y = self.train_df.pop('sepal_length')
        self.test_df = iris[~idx].reset_index(drop=True)

    def is_generate_idempotency(self, atom):
        """atom のべき等チェック"""
        feat_1 = atom.generate(self.train_df, self.y)
        feat_2 = atom.generate(self.train_df)

        return (feat_1 != feat_2).sum().sum() == 0


class TestCountEncodingAtom(BaseTestCase):
    def setup_method(self):
        super(TestCountEncodingAtom, self).setup_method()

        class IrisCountEncodingAtom(CountEncodingAtom):
            use_columns = ['petal_length', 'species']

        self.atom = IrisCountEncodingAtom()

    def test_generate_data(self):
        feat_train = self.atom.generate(self.train_df, self.y)

        assert len(self.train_df) == len(feat_train)

        feat_test = self.atom.generate(self.test_df)
        assert len(self.test_df) == len(feat_test)

    def test_generate_idempotency(self):
        assert self.is_generate_idempotency(self.atom)


class TestOneHotEncodingAtom(BaseTestCase):
    def setup_method(self):
        super(TestOneHotEncodingAtom, self).setup_method()

        class IrisOneHotAtom(OneHotEncodingAtom):
            use_columns = ['species']

        self.atom = IrisOneHotAtom()

    def test_generate(self):
        self.atom.generate(self.train_df, self.y)
        self.atom.generate(self.test_df)

    def test_generate_idempotency(self):
        assert self.is_generate_idempotency(self.atom)


class TestInnerMergeAtom(BaseTestCase):
    def setup_method(self):
        super(TestInnerMergeAtom, self).setup_method()

        class IrisInnerMergeAtom(InnerMergeAtom):
            use_columns = ['petal_length', 'species']

        self.atom_class = IrisInnerMergeAtom

    @pytest.mark.parametrize('agg', [
        'mean', 'std', 'median', 'min', 'max'
    ])
    def test_generate(self, agg):
        atom = self.atom_class(merge_key='species', agg=agg)

        atom.generate(self.train_df, self.y)
        atom.generate(self.test_df)

        assert self.is_generate_idempotency(atom)
