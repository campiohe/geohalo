"""Cache key + filename derivation."""

from geohalo.cache import CACHE_KEY_PREFIX, _cache_key, _local_key_filename


def test_cache_key_format() -> None:
    grid_digest = b"\x01" * 32
    poly_digest = b"\x02" * 32
    key = _cache_key(grid_digest, poly_digest, 2)
    assert key == f"{CACHE_KEY_PREFIX}:0101010101010101:0202020202020202:2"


def test_local_filename_format() -> None:
    grid_digest = b"\x01" * 32
    poly_digest = b"\x02" * 32
    fname = _local_key_filename(grid_digest, poly_digest, 4)
    assert fname == "0101010101010101_0202020202020202_4.pkl"


def test_different_factor_different_key() -> None:
    d1 = b"\x01" * 32
    d2 = b"\x02" * 32
    assert _cache_key(d1, d2, 1) != _cache_key(d1, d2, 2)
