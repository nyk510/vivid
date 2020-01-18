import numpy as np
import pandas as pd
import pytest

from tests.utils import is_close_to_zero
from vivid.featureset.atoms import StringContainsAtom, AbstractAtom, NotMatchLength, AbstractMergeAtom
from vivid.featureset.utils import create_data_loader


def test_string_contains_atom():
    class TitleContainsAtom(StringContainsAtom):
        queryset = {
            'title': ['a', 'b'],
            'place': ['osaka', 'o']
        }

    df = pd.DataFrame([
        ['a', 'a', 'b', 'c', 'b'],
        ['tokyo', 'OSaka', 'osaka', None, 'kobe']
    ], index=['title', 'place']).T

    df_feat = TitleContainsAtom().generate(df)

    assert is_close_to_zero([1, 1, 0, 0, 0], df_feat.values[:, 0])
    assert is_close_to_zero([0, 0, 1, 0, 1], df_feat.values[:, 1])

    assert is_close_to_zero([0, 1, 1, 0, 0], df_feat['place_osaka'])
    assert is_close_to_zero([1, 1, 1, 0, 1], df_feat['place_o'])


class TestAbstractAtom:

    def test_implementation(self):
        df_input = pd.DataFrame(np.random.uniform(size=(10, 4)))

        class Atom1(AbstractAtom):
            use_columns = None

        with pytest.raises(NotImplementedError):
            Atom1().generate(df_input)

        class Atom2(AbstractAtom):
            use_columns = ('hoge', None,)

        with pytest.raises(ValueError):
            Atom2()

        class Atom3(AbstractAtom):
            use_columns = ('hogehoge')  # not tuple

        with pytest.raises(TypeError):
            Atom3()

    def test_input_columns(self):
        df_input = pd.DataFrame(np.random.uniform(size=(10, 2)), columns=['foo', 'bar'])

        class NotMatchAtom(AbstractAtom):
            use_columns = ('foo', 'bar',)

            def transform(self, input_df):
                # return invalid shape datafrme
                return input_df.sample(5)

        with pytest.raises(NotMatchLength):
            NotMatchAtom().generate(df_input)

        class InvalidColumnAtom(AbstractAtom):
            # not exist input dataframe
            use_columns = ('hoge',)

        with pytest.raises(ValueError):
            InvalidColumnAtom().generate(df_input)


MERGE_KEY = 'key'


def create_test_df():
    df_master = pd.DataFrame([
        [1, 2],
        [2, 5],
        [3, 10]
    ], columns=[MERGE_KEY, 'col1'])
    return df_master


test_loader = create_data_loader(create_test_df)


class TestMergeAtom:
    def setup_method(self):
        self.df_not_have_key = pd.DataFrame(np.random.uniform(size=(10, 2)), columns=['foo', 'bar'])
        self.df_has_key = pd.DataFrame([
            [1, 2], [1, 2], [1, 2], [1, 2], [2, 2],
        ], columns=[MERGE_KEY, 'hoge'])

    def test_invalid_implement(self):
        class InvalidAtom(AbstractMergeAtom):
            pass

        with pytest.raises(AttributeError):
            InvalidAtom()

        class InvalidAtom2(AbstractMergeAtom):
            merge_key = MERGE_KEY

            def generate_outer_feature(self):
                return self.df_outer

        with pytest.raises(NotImplementedError):
            InvalidAtom2().generate(self.df_has_key)

    def test_basic(self):
        class BasicMergeAtom(AbstractMergeAtom):
            merge_key = MERGE_KEY

            def read_outer_dataframe(self):
                return test_loader.read()

            def generate_outer_feature(self):
                return self.df_outer

        with pytest.raises(ValueError):
            BasicMergeAtom().generate(self.df_not_have_key)

        df_out = BasicMergeAtom().generate(self.df_has_key)
        assert df_out.columns.tolist() == ['col1']
        assert df_out.shape == (len(self.df_has_key), 1)

    def test_master_loader(self):
        class BasicMergeAtom(AbstractMergeAtom):
            merge_key = MERGE_KEY

            def read_outer_dataframe(self):
                return test_loader.read()

            def generate_outer_feature(self):
                return self.df_outer

        atom = BasicMergeAtom()
        atom.generate(self.df_has_key)

        assert atom._master_dataframe is not None
