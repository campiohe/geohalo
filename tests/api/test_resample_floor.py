import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings, strategies as st

from geohalo.api import resample_grid, resample_grid_with_matrix
from geohalo.geometry import parent_flat_2d
from geohalo.resampler import Resampler


def _da(values, lats, lons, extra_dims=()):
    dims = (*extra_dims, "latitude", "longitude")
    return xr.DataArray(values, dims=dims, coords={"latitude": lats, "longitude": lons})


def _sharp_field():
    """A wet cell surrounded by dry cells: the iteration overshoots negative."""
    lats = np.arange(0.0, 5.0)
    lons = np.arange(0.0, 5.0)
    vals = np.zeros((5, 5))
    vals[2, 2] = 10.0
    return _da(vals, lats, lons), lats, lons


def _block_means(out, src_lats, src_lons):
    p = parent_flat_2d(src_lats, src_lons, out["latitude"].to_numpy(), out["longitude"].to_numpy())
    sums = np.bincount(p, weights=out.to_numpy().ravel(), minlength=src_lats.size * src_lons.size)
    counts = np.bincount(p, minlength=src_lats.size * src_lons.size)
    return (sums / counts).reshape(src_lats.size, src_lons.size)


def test_unfloored_baseline_goes_negative() -> None:
    # guards the premise: if this stops failing-to-be-negative, revisit the feature docs
    da, _, _ = _sharp_field()
    out = resample_grid(da, target_resolution=0.25, iterations=4)
    assert float(out.min()) < 0.0


def test_floor_clamps_adversarial_field() -> None:
    da, _, _ = _sharp_field()
    out = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    assert float(out.min()) >= 0.0


def test_floor_preserves_block_means() -> None:
    da, lats, lons = _sharp_field()
    out = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    np.testing.assert_allclose(_block_means(out, lats, lons), da.values, atol=1e-9)


def test_floor_none_is_bitwise_noop() -> None:
    rng = np.random.default_rng(3)
    da = _da(rng.random((4, 4)), np.arange(4.0), np.arange(4.0))
    base = resample_grid(da, target_resolution=0.5, iterations=2)
    explicit = resample_grid(da, target_resolution=0.5, iterations=2, floor=None)
    np.testing.assert_array_equal(base.values, explicit.values)


def test_floor_with_matrix_matches_resample_grid() -> None:
    da, lats, lons = _sharp_field()
    t_lat = np.arange(0.0, 4.0 + 0.125, 0.25)
    t_lon = np.arange(0.0, 4.0 + 0.125, 0.25)
    r = Resampler.compute(lats, lons, t_lat, t_lon, iterations=4)
    out = resample_grid_with_matrix(da, r, floor=0.0)
    expected = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    np.testing.assert_allclose(out.values, expected.values, atol=1e-12)


def test_mapping_floor_on_dataarray_raises() -> None:
    da, _, _ = _sharp_field()
    with pytest.raises(TypeError, match="Dataset"):
        resample_grid(da, target_resolution=0.25, floor={"tp": 0.0})


def _sharp_dataset():
    da, lats, lons = _sharp_field()
    temp = _da(np.full((5, 5), 280.0), lats, lons)
    ds = xr.Dataset({"tp": da, "t2m": temp, "scalar_meta": 1.0})
    return ds, lats, lons


def test_dataset_float_floors_all_spatial_vars() -> None:
    ds, _, _ = _sharp_dataset()
    out = resample_grid(ds, target_resolution=0.25, iterations=4, floor=0.0)
    assert float(out["tp"].min()) >= 0.0
    assert float(out["t2m"].min()) >= 0.0
    assert "scalar_meta" in out  # non-spatial vars pass through untouched


def test_dataset_mapping_floors_named_var_only() -> None:
    ds, _, _ = _sharp_dataset()
    out = resample_grid(ds, target_resolution=0.25, iterations=4, floor={"tp": 0.0})
    base = resample_grid(ds, target_resolution=0.25, iterations=4)
    assert float(out["tp"].min()) >= 0.0
    np.testing.assert_array_equal(out["t2m"].values, base["t2m"].values)  # untouched, bit-for-bit


def test_dataset_mapping_unknown_var_raises() -> None:
    ds, _, _ = _sharp_dataset()
    with pytest.raises(ValueError, match="not spatial data vars"):
        resample_grid(ds, target_resolution=0.25, floor={"typo": 0.0})


def test_dataset_mapping_nonspatial_var_raises() -> None:
    ds, _, _ = _sharp_dataset()
    with pytest.raises(ValueError, match="not spatial data vars"):
        resample_grid(ds, target_resolution=0.25, floor={"scalar_meta": 0.0})


def test_floor_descending_lats_match_ascending() -> None:
    da, _, _ = _sharp_field()
    out_asc = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    out_desc = resample_grid(
        da.sortby("latitude", ascending=False), target_resolution=0.25, iterations=4, floor=0.0,
    )
    np.testing.assert_allclose(out_desc.sortby("latitude").values, out_asc.values, atol=1e-12)


def test_floor_batched_matches_per_slice() -> None:
    da, lats, lons = _sharp_field()
    batched = _da(np.stack([da.values, 2.0 * da.values]), lats, lons, extra_dims=("member",))
    out = resample_grid(batched, target_resolution=0.25, iterations=4, floor=0.0)
    slice0 = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    np.testing.assert_allclose(out.isel(member=0).values, slice0.values, atol=1e-12)
    assert float(out.min()) >= 0.0


def test_floor_nan_footprint_matches_linear_path() -> None:
    da, _, _ = _sharp_field()
    da = da.copy()
    da.values[0, 0] = np.nan
    base = resample_grid(da, target_resolution=0.25, iterations=4)
    out = resample_grid(da, target_resolution=0.25, iterations=4, floor=0.0)
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(base.values))


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=25, deadline=None)
def test_property_floored_output_nonneg_and_mean_preserving(seed: int) -> None:
    rng = np.random.default_rng(seed)
    lats = np.arange(0.0, 3.0)
    lons = np.arange(0.0, 4.0)
    # sparse non-negative field: many exact zeros next to positive cells
    vals = np.maximum(rng.normal(0.0, 1.0, size=(3, 4)), 0.0)
    da = _da(vals, lats, lons)
    out = resample_grid(da, target_resolution=0.25, iterations=3, floor=0.0)
    assert float(out.min()) >= 0.0
    np.testing.assert_allclose(_block_means(out, lats, lons), vals, atol=1e-9)
