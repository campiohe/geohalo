import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

from geohalo.geometry import cell_areas
from geohalo.resampler import Resampler


def nested_grids():
    return np.arange(40.25, 60., .5), np.arange(.25, 20., .5), np.arange(41., 60., 2.), np.arange(1., 20., 2.)


@pytest.mark.parametrize("descending_lat", [False, True])
@pytest.mark.parametrize("descending_lon", [False, True])
@pytest.mark.parametrize("reverse_target", [False, True])
def test_issue_15_physical_block_means_and_mass(descending_lat, descending_lon, reverse_target):
    lat, lon, tlat, tlon = nested_grids()
    source = np.random.default_rng(0).random((lat.size, lon.size)) * 10
    area = cell_areas(lat, lon)
    exact = (source * area).reshape(10, 4, 10, 4).sum((1, 3)) / area.reshape(10, 4, 10, 4).sum((1, 3))
    total_mass = np.sum(source * area)
    if descending_lat:
        lat, source = lat[::-1], source[::-1]
    if descending_lon:
        lon, source = lon[::-1], source[:, ::-1]
    if reverse_target:
        tlat, tlon, exact = tlat[::-1], tlon[::-1], exact[::-1, ::-1]
    operator = Resampler.compute(lat, lon, tlat, tlon, method="conservative")
    result = operator.apply_grid(source, descending=descending_lat).reshape(10, 10)
    np.testing.assert_allclose(result, exact, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.sum(result * abs(cell_areas(tlat, tlon))), total_mass, rtol=2e-15)
    assert operator.method == "conservative"
    assert operator.transform_matrix is None
    np.testing.assert_allclose(operator.coverage, 1.)
    assert "axis_nnz=" in repr(operator)


def test_issue_15_storm_stays_bounded_and_mass_conserved():
    fine_lat, fine_lon, lat, lon = nested_grids()
    rain = np.zeros((10, 10))
    rain[4, 4] = 40.
    operator = Resampler.compute(lat, lon, fine_lat, fine_lon, method="conservative")
    out = operator.apply_grid(rain).reshape(40, 40)
    np.testing.assert_array_equal(out, np.repeat(np.repeat(rain, 4, axis=0), 4, axis=1))
    np.testing.assert_allclose(np.sum(out * cell_areas(fine_lat, fine_lon)),
                               np.sum(rain * cell_areas(lat, lon)), rtol=2e-15)


@pytest.mark.parametrize("period", [None, 360])
@pytest.mark.parametrize(("n_source", "n_target"), [(7, 11), (11, 7)])
def test_nonnested_arbitrary_ratios_preserve_mass_and_bounds(period, n_source, n_target):
    source_bounds = np.linspace(-90, 90, n_source + 1), np.linspace(-180, 180, n_source + 2)
    target_bounds = np.linspace(-90, 90, n_target + 1), np.linspace(-180, 180, n_target + 2)
    slat, slon = ((axis[:-1] + axis[1:]) / 2 for axis in source_bounds)
    tlat, tlon = ((axis[:-1] + axis[1:]) / 2 for axis in target_bounds)
    operator = Resampler.compute(slat, slon, tlat, tlon, method="conservative", period=period,
                                 source_bounds=source_bounds, target_bounds=target_bounds)
    values = np.random.default_rng(0).uniform(1., 10., size=(slat.size, slon.size))
    result = operator.apply_grid(values).reshape(tlat.size, tlon.size)
    source_area = np.diff(np.sin(np.deg2rad(source_bounds[0])))[:, None] * np.diff(source_bounds[1])[None, :]
    target_area = np.diff(np.sin(np.deg2rad(target_bounds[0])))[:, None] * np.diff(target_bounds[1])[None, :]
    np.testing.assert_allclose(np.sum(values * source_area), np.sum(result * target_area), rtol=2e-15)
    assert result.min() >= values.min() - 1e-14
    assert result.max() <= values.max() + 1e-14
    np.testing.assert_allclose(operator.coverage, 1., atol=1e-14)


def partial_operator(normalization="destination"):
    return Resampler.compute(
        np.array([0.]), np.array([0., 1.]), np.array([0.]), np.arange(-1.5, 3., 1.),
        method="conservative", normalization=normalization,
        source_bounds=(np.array([-.5, .5]), np.array([-.5, .5, 1.5])),
        target_bounds=(np.array([-.5, .5]), np.arange(-2., 4.)),
    )


@pytest.mark.parametrize("normalization", ["destination", "covered"])
@pytest.mark.parametrize("skipna", [False, True])
def test_partial_unmapped_and_missing_semantics(normalization, skipna):
    operator = partial_operator(normalization)
    values = np.array([[[2., 4.]], [[2., np.nan]], [[np.nan, np.nan]]])
    actual = operator.apply_grid(values, skipna=skipna)
    covered = normalization == "covered" or skipna
    expected = [[np.nan, 2. if covered else 1., 3., 4. if covered else 2., np.nan],
                [np.nan, 2. if covered else 1., 2. if skipna else np.nan, np.nan, np.nan],
                [np.nan] * 5]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_array_equal(operator.coverage, [[0., .5, 1., .5, 0.]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.complex64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0, 2)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran"])
@pytest.mark.parametrize("refine", [False, True])
def test_batched_layouts_and_dtypes(dtype, batch_shape, layout, refine):
    lat, lon = np.arange(4.), np.arange(6.)
    target_lat = np.arange(-.25, 3.75, .5) if refine else lat
    target_lon = np.arange(-.25, 5.75, .5) if refine else lon
    operator = Resampler.compute(lat, lon, target_lat, target_lon, method="conservative")
    values = np.arange(np.prod((*batch_shape, 4, 6))).reshape(*batch_shape, 4, 6).astype(dtype)
    if layout == "strided":
        values = values[..., ::-1]
    if layout == "fortran":
        values = np.asfortranarray(values)
    actual = operator.apply_grid(values)
    expected = np.repeat(np.repeat(values, 2, axis=-2), 2, axis=-1) if refine else values
    np.testing.assert_array_equal(actual, expected.reshape(*batch_shape, target_lat.size * target_lon.size))
    assert actual.dtype == np.result_type(dtype, np.float64)


def test_refinement_projects_contiguous_arrays_for_normalization():
    operator = Resampler.compute(
        np.arange(4.), np.arange(6.), np.linspace(0, 3, 10), np.linspace(0, 5, 15),
        method="conservative",
    )
    values = np.random.default_rng(0).random((4, 6))
    total = operator._conservative_grid._project(values)
    a, b = operator.axis_weights
    np.testing.assert_allclose(total, a.toarray() @ values @ b.toarray().T, rtol=2e-15)
    assert total.flags.c_contiguous


@pytest.mark.parametrize("normalization", ["destination", "covered"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("period", [None, 360])
def test_nonnested_refinement_matches_full_overlap_operator(normalization, skipna, descending, period):
    source_bounds = np.linspace(-60, 60, 5), np.linspace(-180, 180, 7)
    target_bounds = np.linspace(-75, 75, 11), np.linspace(-225, 135, 16)
    lat, lon = ((edges[:-1] + edges[1:]) / 2 for edges in source_bounds)
    tlat, tlon = ((edges[:-1] + edges[1:]) / 2 for edges in target_bounds)
    if descending:
        lat, tlat = lat[::-1], tlat[::-1]
        source_bounds = source_bounds[0][::-1], source_bounds[1]
        target_bounds = target_bounds[0][::-1], target_bounds[1]
    operator = Resampler.compute(
        lat, lon, tlat, tlon, method="conservative", normalization=normalization,
        period=period, source_bounds=source_bounds, target_bounds=target_bounds,
    )
    values = np.random.default_rng(0).uniform(270, 290, (3, 4, 6)).astype(np.float32)
    values[1, 1:3, 1:3] = np.nan
    values[2] = np.nan
    overlap = sp.kron(*operator.axis_weights, format="csr")
    coverage = np.asarray(overlap.sum(axis=1)).ravel()
    expected = np.full((3, tlat.size * tlon.size), np.nan)
    for index, step in enumerate(values):
        flat = step.ravel()
        if skipna:
            valid = ~np.isnan(flat)
            total = overlap @ np.where(valid, flat, 0)
            denominator = overlap @ valid.astype(np.float64)
        else:
            total, denominator = overlap @ flat, coverage
        if skipna or normalization == "covered":
            np.divide(total, denominator, out=expected[index], where=denominator > 0)
        else:
            expected[index] = np.where(coverage > 0, total, np.nan)
    actual = operator.apply_grid(values[:, ::-1] if descending else values, descending=descending, skipna=skipna)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, equal_nan=True)


def test_two_pass_masking_normalizes_only_after_both_axes():
    lat, lon = np.array([0., 1.]), np.array([0., 1.])
    target = np.array([.5])
    bounds = (np.array([-.5, 1.5]), np.array([-.5, 1.5]))
    operator = Resampler.compute(lat, lon, target, target, method="conservative", target_bounds=bounds)
    values = np.array([[0., 10.], [100., np.nan]])
    area = cell_areas(lat, lon)
    expected = np.nansum(values * area) / area[~np.isnan(values)].sum()
    np.testing.assert_allclose(operator.apply_grid(values, skipna=True), [expected])


@pytest.mark.parametrize("target_shape", [(4, 30), (30, 4)])
def test_no_kronecker_and_both_contraction_orders(target_shape, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("conservative regridding must never form a Kronecker matrix")

    monkeypatch.setattr(sp, "kron", forbidden)
    lat, lon = np.linspace(0, 10, 12), np.linspace(0, 20, 15)
    operator = Resampler.compute(lat, lon, np.linspace(1, 9, target_shape[0]),
                                 np.linspace(1, 19, target_shape[1]), method="conservative")
    values = np.random.default_rng(0).random((12, 15))
    a, b = operator.axis_weights
    expected = a.toarray() @ values @ b.toarray().T
    np.testing.assert_allclose(operator.apply_grid(values).reshape(target_shape), expected, rtol=2e-15)


@pytest.mark.parametrize("skipna", [False, True])
def test_large_batch_memory_is_bounded_per_slice(skipna):
    lat, lon = np.linspace(-80, 80, 201), np.linspace(-170, 170, 251)
    operator = Resampler.compute(lat, lon, lat[::2], lon[::2], method="conservative")
    values = np.ones((24, lat.size, lon.size), dtype=np.float32)
    if skipna:
        values[:, 50, 50] = np.nan
    for _ in range(2):
        tracemalloc.start()
        try:
            result = operator.apply_grid(values, descending=True, skipna=skipna)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < result.nbytes + 12 * values[0].nbytes


@pytest.mark.parametrize("coarsen", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_mixed_axis_ratios_avoid_large_intermediates(coarsen, skipna):
    source_shape, target_shape = (400, 40), (20, 1600)
    if coarsen:
        source_shape, target_shape = target_shape, source_shape
    lat, lon = np.linspace(-80, 80, source_shape[0]), np.linspace(-170, 170, source_shape[1])
    tlat, tlon = np.linspace(-79, 79, target_shape[0]), np.linspace(-169, 169, target_shape[1])
    operator = Resampler.compute(lat, lon, tlat, tlon, method="conservative")
    values = np.ones(source_shape)
    if skipna:
        values[::37, ::13] = np.nan
    tracemalloc.start()
    try:
        result = operator.apply_grid(values, skipna=skipna)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The small intermediate has 800 cells; the other order needs 640,000.
    assert peak < 8 * (values.nbytes + result.nbytes)
    if skipna:
        np.testing.assert_allclose(result[np.isfinite(result)], 1., rtol=2e-15)
    else:
        np.testing.assert_allclose(result, operator.coverage.ravel(), rtol=2e-15)


def test_bad_value_shape_and_nonconservative_options():
    operator = partial_operator()
    with pytest.raises(ValueError, match="trailing source shape"):
        operator.apply_grid(np.ones((2, 2)))
    coords = np.arange(2.)
    classic = Resampler.compute(coords, coords, coords, coords)
    assert classic.method == "meanpreserving"
    with pytest.raises(ValueError, match="only available"):
        _ = classic.coverage
    with pytest.raises(ValueError, match="requires method"):
        classic.apply_grid(np.ones((2, 2)), skipna=True)
