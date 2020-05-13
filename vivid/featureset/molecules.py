import os
from typing import List

import joblib
import pandas as pd

from vivid.core import AbstractFeature
from vivid.utils import timer
from .atoms import AbstractAtom


class Molecule:
    def __init__(self, atoms: List[AbstractAtom], name=None):
        """

        Args:
            atoms(list[AbstractAtom]): 使う atom の配列.
            name(str): この atoms set を表す命名.
        """
        self.atoms = atoms
        self.name = name

    def generate(self, df_input, y=None):
        return pd.concat([atom.generate(df_input, y) for atom in self.atoms], axis=1)


class MoleculeFactory(object):
    molecules = []

    @classmethod
    def create_molecule(cls, atoms, name=None):
        m = Molecule(atoms, name)
        cls.molecules.append(m)
        return m

    @classmethod
    def find(cls, name):
        return list(filter(lambda m: m.name == name, cls.molecules))


create_molecule = MoleculeFactory.create_molecule
find_molecule = MoleculeFactory.find


class MoleculeFeature(AbstractFeature):
    def __init__(self, molecule, parent=None, root_dir=None):
        super(MoleculeFeature, self).__init__(name=molecule.name, parent=parent, root_dir=root_dir)
        self.molecule = molecule

    @property
    def molecule_path(self):
        if self.has_output_dir:
            return os.path.join(self.output_dir, 'molecule.job')
        return None

    def load_molecule(self):
        if not self.is_recording:
            return

        self.molecule = joblib.load(self.molecule_path)
        return

    def call(self, df_source, y=None, test=False):
        if test:
            self.load_molecule()

        out_df = pd.DataFrame()

        for atom in self.molecule.atoms:
            with timer(self.logger, format_str=f'{str(atom)} ' + '{:.3f}[s]'):
                out_df = pd.concat([out_df, atom.generate(df_source, y)], axis=1)

        if not test and self.is_recording:
            joblib.dump(self.molecule, self.molecule_path)

        return out_df
