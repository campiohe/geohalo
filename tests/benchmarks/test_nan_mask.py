import numpy as np
import pytest
import xarray as xr

from benchmarks._data import apply_nan_mask


def _field() -> xr.DataArray:
    # (member, step, latitude, longitude)
    return xr.DataArray(
        np.ones((2, 3, 4, 5)),
        dims=("member", "step", "latitude", "longitude"),
    )


def test_apply_nan_mask_leaves_original_unmodified() -> None:
    da = _field()
    apply_nan_mask(da, fraction=0.5, seed=0)
    assert not np.isnan(da.to_numpy()).any()


def test_apply_nan_mask_masks_same_cells_across_batch() -> None:
    out = apply_nan_mask(_field(), fraction=0.5, seed=0)
    nan_mask = np.isnan(out.to_numpy())
    ref = nan_mask[0, 0]
    assert (nan_mask == ref).all()  # every (member, step) shares one spatial mask


def test_apply_nan_mask_fraction_in_range() -> None:
    out = apply_nan_mask(_field(), fraction=0.5, seed=0)
    frac = np.isnan(out.to_numpy()[0, 0]).mean()
    assert 0.2 < frac < 0.8  # 20 cells, seeded; loose bound around 0.5


def test_apply_nan_mask_rejects_bad_fraction() -> None:
    da = xr.DataArray(np.ones((2, 2)), dims=("latitude", "longitude"))
    with pytest.raises(ValueError, match="fraction"):
        apply_nan_mask(da, fraction=1.0)
