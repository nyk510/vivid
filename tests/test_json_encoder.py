from vivid import json_encoder
import pytest
import pandas as pd
import numpy as np


class Hoge:
    pass


class HasRepr:
    def repr_json(self):
        return 'foo-foo'


@pytest.mark.parametrize('input, expect', [
    (np.array([1, 2, 3]), [1, 2, 3]),
    (pd.Index([1, 2, 3]), [1, 2, 3]),
    (pd.DataFrame([[1, 2, 3]], ), {0: {0: 1}, 1: {0: 2}, 2: {0: 3}}),
    (pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='foo'), {'a': 1, 'b': 2, 'c': 3}),
    (pd.to_datetime('2020-05-14'), '2020-05-14 00:00:00'),
    (Hoge, str(Hoge)),
    (HasRepr(), 'foo-foo'),
    (1 + 2j, '(1+2j)'),  # complex number

    (Exception('foo'), str(Exception('foo'))),  # exception class
    (Hoge(), 'Hoge not JSON serializable'),  # no filter match
])
def test_encoder_preprocess_function(input, expect):
    encoder = json_encoder.NestedEncoder()
    out = encoder.default(input)
    assert out == expect, out
