import numpy as np
import pandas as pd

from vivid.featureset import AbstractAtom
from vivid.featureset import create_molecule, find_molecule
from vivid.featureset.molecules import MoleculeFeature
from .factories import generate_price_dataframe


class PricePlusIdAtom(AbstractAtom):
    use_columns = ('price', 'id',)

    def __init__(self):
        super(PricePlusIdAtom, self).__init__()
        self.mean = None

    def get_mean(self, y=None):
        if y is not None:
            if y is None:
                raise ValueError()
            m = np.mean(y)
            self.mean = m
        return self.mean

    def fit(self, input_df: pd.DataFrame, y=None):
        self.get_mean(y)
        return self

    def transform(self, input_df):
        df_out = pd.DataFrame()
        df_out['price_plus_id'] = input_df['price'] + input_df['id']
        df_out['y_mean'] = self.get_mean()

        return df_out


class TestMolecule:
    def test_simple_molecule(self):
        """atom """
        input_df = generate_price_dataframe()
        m1 = create_molecule([PricePlusIdAtom()], name='sample1')
        m2 = create_molecule([PricePlusIdAtom(), PricePlusIdAtom()], name='sample2')

        for m in [m1, m2]:
            m.generate(input_df)

        assert find_molecule('sample1')[0] == m1

    def test_molecule_feature_combined(self):
        input_df = generate_price_dataframe()

        class PreprocessAtom(AbstractAtom):
            use_columns = ('price',)

            def transform(self, input_df):
                out_df = input_df.copy()
                out_df['price'] = np.where(input_df['price'] > 1, 1, 0)
                return out_df

        preprocess_molecule = create_molecule([PreprocessAtom()], name='preprocess')
        m1 = create_molecule([PricePlusIdAtom()], name='m1')

        feat_preprocess = MoleculeFeature(preprocess_molecule)
        feat_1 = MoleculeFeature(m1, parent=feat_preprocess)

        feat_df = feat_1.fit(input_df, y=input_df.values[:, 0])
        feat_reproduct_df = feat_1.predict(input_df)
        assert (feat_df != feat_reproduct_df).sum().sum() == 0
