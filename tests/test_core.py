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
