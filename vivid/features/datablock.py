import os
from typing import Tuple, Union

import pandas as pd

from vivid.core import BaseBlock


class DataBlock(BaseBlock):
    def __init__(self, name: str):
        super(DataBlock, self).__init__(name=name, parent=None)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class CSVBlock(DataBlock):
    def __init__(self, path_to_csv: str):
        self.path_to_csv = path_to_csv
        super(CSVBlock, self).__init__(name=os.path.basename(path_to_csv))

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        return pd.read_csv(self.path_to_csv)


class RelationBlock(BaseBlock):
    def __init__(self,
                 name: str,
                 key: Union[str, Tuple[str, str]],
                 parent_from: BaseBlock,
                 parent_to: BaseBlock):
        super(RelationBlock, self).__init__(name=name, parent=[parent_from, parent_to])

        self.parent_from = parent_from
        self.parent_to = parent_to

        self._key = key

        if isinstance(key, str):
            keys = [key, key]
        else:
            try:
                if len(key) != 2:
                    raise ValueError('key length must be 2.')
                else:
                    keys = key
            except TypeError as e:
                raise ValueError(f'key is not iterable. {e}')

        self.keys = keys

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        cols = source_df.columns  # type: pd.Index

        from_col_idx = cols.str.startswith(self.parent_from.name)
        to_col_idx = cols.str.startswith(self.parent_to.name)

        from_key, to_key = self.keys
        from_key = f'{self.parent_from.name}_{from_key}'
        to_key = f'{self.parent_to.name}_{to_key}'

        from_df = source_df.iloc[:, from_col_idx]
        to_df = source_df.iloc[:, to_col_idx]

        if from_key not in from_df.columns:
            raise ValueError()

        if to_key not in to_df.columns:
            raise ValueError('must be {}'.format(','.join(to_df.columns)))

        # from dataframe key must be unique
        if from_df[from_key].value_counts().max() > 1:
            raise ValueError(f'duplicated key: {from_key}')

        out_df = pd.merge(to_df, from_df, left_on=to_key, right_on=from_key, how='left')
        out_df = out_df.drop(columns=[from_key])
        return out_df
