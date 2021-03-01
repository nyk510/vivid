import pytest

from vivid.backends import LocalExperimentBackend
from vivid.core import BaseBlock, network_hash


def test_network_hash():
    a = BaseBlock('a')
    b = BaseBlock('b')
    assert network_hash(a) != network_hash(b)
    assert network_hash(a) == network_hash(a)

    c = BaseBlock('c', parent=[a, b])
    hash1 = network_hash(c)
    a._parent = [BaseBlock('z')]
    hash2 = network_hash(c)
    assert hash1 != hash2


class CustomBlock(BaseBlock):
    """fronzen / unzip のためのテストクラス"""
    _save_attributes = [
        '_hoge'
    ]

    def __init__(self):
        super(CustomBlock, self).__init__('test')
        self._hoge = {
            'foo': 'bar'
        }


def test_block_frozen_and_unzip_attribute(tmpdir):
    block = CustomBlock()

    exp = LocalExperimentBackend(to=tmpdir)
    block.frozen(exp)
    del block._hoge

    block.unzip(exp)
    assert block._hoge is not None


def test_set_attributes_not_in_current_attribute(tmpdir):
    class Test1(CustomBlock):
        _save_attributes = [
            'bar'
        ]

    exp = LocalExperimentBackend(tmpdir)

    block = Test1()

    with pytest.warns(UserWarning, match='field `bar` is not found in the block `test`.'):
        block.frozen(exp)


def test_cant_load_from_experiment(tmpdir):
    exp = LocalExperimentBackend(tmpdir)
    block = CustomBlock()

    with pytest.warns(UserWarning, match='Fail to load'):
        block.unzip(exp)
