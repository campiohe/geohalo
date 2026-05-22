"""GridSpec digest behavior."""

import numpy as np

from geohalo.grid import GridSpec


def test_digest_collides_for_ascending_and_descending() -> None:
    """CLAUDE.md invariant #1: digest is computed after normalization,
    so ascending and descending inputs hash to the same value.
    """
    asc = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([10.0, 11.0]))
    desc = GridSpec(lats=np.array([2.0, 1.0, 0.0]), lons=np.array([10.0, 11.0]))
    assert asc.digest == desc.digest


def test_digest_changes_with_lat_values() -> None:
    a = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    b = GridSpec(lats=np.array([0.0, 2.0]), lons=np.array([0.0, 1.0]))
    assert a.digest != b.digest


def test_digest_changes_with_lon_values() -> None:
    a = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    b = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 2.0]))
    assert a.digest != b.digest


def test_digest_is_32_bytes() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    assert isinstance(grid.digest, bytes)
    assert len(grid.digest) == 32  # SHA-256
