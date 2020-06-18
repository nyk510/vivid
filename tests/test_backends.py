import os

import pytest

from vivid.backends.dataframes import CSVBackend, JoblibBackend, FeatherBackend, DataFrameBackend


@pytest.mark.parametrize('name,expect', [
    ('my-file', 'my-file.csv'),
    ('user.hoge', 'user.csv'),
    ('/path/to/myfile.foo', '/path/to/myfile.csv'),
    ('/foo--/path/to/myfile.foo', '/foo--/path/to/myfile.csv'),
])
def test_filename(name, expect):
    backend = DataFrameBackend()
    backend.ext = 'csv'
    assert backend.to_filename(name) == expect


@pytest.mark.parametrize('backend_class,', [
    JoblibBackend, FeatherBackend, CSVBackend,
])
def test_backend(backend_class, output_dir, toy_df):
    backend = backend_class()

    path = os.path.join(output_dir, 'boston.ff')
    os.makedirs(output_dir, exist_ok=True)
    
    backend.save(toy_df, path)

    df2 = backend.load(path)
    assert toy_df.equals(df2)


def test_invalid_ext_warning():
    class InvalidExtBackend(DataFrameBackend):
        ext = '.foo'

    backend = InvalidExtBackend()

    with pytest.warns(UserWarning) as records:
        backend.get_ext()

    assert len(records) == 1
