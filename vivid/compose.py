from typing import List

import numpy as np
import pandas as pd

from .core import AbstractFeature
from .featureset.molecules import find_molecule, MoleculeFeature


class TrainComposer:
    def __init__(self, root_dir, molecule='basic', ):
        if isinstance(molecule, str):
            molecule = find_molecule(molecule)[0]
        self.entry_point = MoleculeFeature(molecule,
                                           root_dir=root_dir)
        self.models = None  # type: List[AbstractFeature]

    def build(self, stacking=False, *args, **kwargs):
        return

    def fit(self, input_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        out_df = pd.DataFrame()
        for model in self.models:
            pred_df = model.fit(input_df, y, force=True)
            out_df[model.name] = pred_df.values[:, 0]

        return out_df

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        for m in self.models:
            pred_df = m.predict(input_df)
            out_df[m.name] = pred_df.values[:, 0]

        return out_df
