import subprocess
import sys
import weakref
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from geohalo import LocalCache, RestrictedOperator, reduce_with_operator
from tests.restricted_operator._helpers import make_grid, make_operator


def _windows(plan, values):
    return [values[..., rows, cols] for rows, cols in plan.windows]


def _manual_gather(plan, arrays):
    return np.concatenate([
        values.reshape((*values.shape[:-2], -1))[..., positions]
        for values, positions in zip(arrays, plan.gathers, strict=True)
    ], axis=-1)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (5,), (3, 2), (0, 2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_array_methods_match_full_grid(descending, batch_shape, dtype):
    operator = make_operator(multiindex=True)
    grid = make_grid(operator, descending=descending, batch_shape=batch_shape, dtype=dtype)
    plan = RestrictedOperator.compute(operator, grid.latitude.values, (3, 1, 4), (2, 3, 1, 4))
    arrays = _windows(plan, grid.values)
    snapshots = [array.copy() for array in arrays]
    gathered = plan.gather(iter(arrays))
    # Explicit sizes support empty batch dimensions, unlike reshape(..., -1).
    expected = np.concatenate([
        array.reshape((*batch_shape, array.shape[-2] * array.shape[-1]))[..., positions]
        for array, positions in zip(arrays, plan.gathers, strict=True)
    ], axis=-1)
    np.testing.assert_array_equal(gathered, expected)
    assert gathered.dtype == dtype
    for how in ("mean", "sum"):
        actual = plan.apply(gathered, how=how)
        expected = reduce_with_operator(grid, operator, how=how).values
        np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(gathered, plan.gather(arrays), strict=True)
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_mixed_window_dtypes_are_promoted_without_changing_values():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    arrays = _windows(plan, make_grid(operator).values)
    assert len(arrays) > 1
    arrays[0] = arrays[0].astype(np.int16)
    arrays[-1] = arrays[-1].astype(np.float64) + 0.125
    actual = plan.gather(arrays)
    np.testing.assert_array_equal(actual, _manual_gather(plan, arrays), strict=True)


def test_window_iterator_does_not_retain_previous_arrays():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    grid = make_grid(operator)
    references = []

    def read():
        for rows, cols in plan.windows:
            assert all(ref() is None for ref in references)
            window = grid.values[..., rows, cols].copy()
            references.append(weakref.ref(window))
            yield window
            del window

    gathered = plan.gather(read())
    assert all(ref() is None for ref in references)
    np.testing.assert_array_equal(plan.apply(gathered), reduce_with_operator(grid, operator).values)


def test_gather_accepts_strided_windows_and_batch_dimensions():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    arrays = _windows(plan, make_grid(operator, batch_shape=(3, 2)).values)

    strided = [array[::-1, ::-1, ::-1, ::-1].swapaxes(0, 1) for array in arrays]
    assert all(not array.flags.c_contiguous for array in strided)
    np.testing.assert_array_equal(plan.gather(strided), _manual_gather(plan, strided))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apply_preserves_coefficient_order_and_mean_promotion(dtype):
    matrix = sp.csr_matrix((np.array([1e16, -1e16, 1], dtype=dtype), [65, 1, 3], [0, 3]), shape=(1, 80))
    operator = replace(make_operator(matrix), row_sums=np.array([3.0], dtype=np.float64))
    plan = RestrictedOperator.compute(operator, operator.source_lat[::-1], 2, 2)
    gathered = plan.gather(_windows(plan, np.ones((2, 8, 10), dtype=dtype)))
    totals = plan.apply(gathered, how="sum")
    means = plan.apply(gathered)
    np.testing.assert_array_equal(totals, np.ones((2, 1), dtype=dtype), strict=True)
    np.testing.assert_array_equal(means, np.full((2, 1), 1 / 3, dtype=np.float64), strict=True)


def test_apply_accepts_strided_gathered_arrays():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    grid = make_grid(operator)
    gathered = plan.gather(_windows(plan, grid.values))
    # Reverse and transpose batch dims; also give the cell dimension a stride.
    padded = np.empty((*gathered.shape[:-1], 2 * gathered.shape[-1]), dtype=gathered.dtype)
    padded[..., ::2] = gathered
    strided = padded[::-1, ::-1, ::2].swapaxes(0, 1)
    assert not strided.flags.c_contiguous
    for how in ("mean", "sum"):
        expected = plan.apply(gathered, how=how)[::-1, ::-1].swapaxes(0, 1)
        np.testing.assert_array_equal(plan.apply(strided, how=how), expected, strict=True)


def test_nan_behavior_is_unchanged():
    operator = make_operator()
    grid = make_grid(operator)
    grid.values[..., 0, 1] = np.nan
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    gathered = plan.gather(_windows(plan, grid.values))
    for how in ("mean", "sum"):
        expected = reduce_with_operator(grid, operator, how=how).values
        actual = plan.apply(gathered, how=how)
        np.testing.assert_array_equal(actual, expected)
        assert np.isnan(actual[..., 0]).all()
        assert np.isfinite(actual[..., 1:]).all()


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3), (0, 2)])
def test_empty_plan(batch_shape):
    operator = make_operator(sp.csr_matrix((3, 80)))
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    gathered = plan.gather([])
    np.testing.assert_array_equal(gathered, np.empty(0), strict=True)
    np.testing.assert_array_equal(plan.apply(gathered), np.zeros(3))
    for how in ("mean", "sum"):
        actual = plan.apply(np.empty((*batch_shape, 0), dtype=np.float32), how=how)
        np.testing.assert_array_equal(actual, np.zeros((*batch_shape, 3)), strict=True)
    with pytest.raises(ValueError, match="expected 0 window arrays"):
        plan.gather([np.zeros((1, 1))])


def test_gather_rejects_incorrect_count_spatial_shape_and_batch_shape():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    arrays = _windows(plan, make_grid(operator).values)
    with pytest.raises(ValueError, match="window arrays"):
        plan.gather(arrays[:-1])
    with pytest.raises(ValueError, match="window arrays"):
        plan.gather([*arrays, arrays[0]])
    for bad in (np.array(1), np.zeros(2), arrays[0][..., :-1, :], arrays[0][..., :, :-1]):
        with pytest.raises(ValueError, match="window 0 expected trailing shape"):
            plan.gather([bad, *arrays[1:]])
    with pytest.raises(ValueError, match="window 1 expected batch shape"):
        plan.gather([arrays[0], arrays[1][0], *arrays[2:]])


def test_apply_rejects_incorrect_shape_and_how():
    operator = make_operator()
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    for bad in (np.array(1), np.empty(plan.matrix.shape[1] + 1), np.empty((3, 0))):
        with pytest.raises(ValueError, match="expected trailing cell dimension"):
            plan.apply(bad)
    with pytest.raises(ValueError, match="how must be 'mean' or 'sum'"):
        plan.apply(np.empty(plan.matrix.shape[1]), how="median")


def test_cached_plan_supports_array_methods(tmp_path):
    operator = make_operator()
    grid = make_grid(operator, descending=True)
    cache = LocalCache(tmp_path)
    cache.get_or_compute_restricted_operator(operator, grid.latitude.values, 3, 4)
    plan = cache.get_or_compute_restricted_operator(operator, grid.latitude.values, 3, 4)
    actual = plan.apply(plan.gather(_windows(plan, grid.values)))
    np.testing.assert_array_equal(actual, reduce_with_operator(grid, operator).values)


def test_array_methods_work_without_dask():
    # Start before importing geohalo so cached modules cannot hide a dependency.
    code = """
import importlib.abc
import importlib.util
import sys

class NoDask(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dask" or fullname.startswith("dask."):
            raise ModuleNotFoundError("Dask is unavailable", name=fullname)

sys.meta_path.insert(0, NoDask())
# An absent optional package is discoverable as None, but any attempted import
# must still fail. Simulate both even when Dask is installed for other tests.
original_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    if name == "dask" or name.startswith("dask."):
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec
import numpy as np
import pandas as pd
import scipy.sparse as sp
from geohalo import ReduceOperator, RestrictedOperator

op = ReduceOperator(sp.csr_matrix([[1., 0., 0., 1.]]), np.array([2.]),
                    pd.Index(["zone"]), np.arange(2.), np.arange(2.), 1, b"test")
plan = RestrictedOperator.compute(op, op.source_lat, 1, 1)
gathered = plan.gather([np.array([[2.]]), np.array([[4.]])])
np.testing.assert_array_equal(plan.apply(gathered), [3.])
assert not any(name == "dask" or name.startswith("dask.") for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
