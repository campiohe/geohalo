# Floored Mean-Preserving Downscaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `floor=` to `resample_grid`/`resample_grid_with_matrix` so non-negative variables (precipitation) never go below a bound, while each source cell's child mean is still preserved exactly.

**Architecture:** The linear transform `T`, its digest, and all cache payloads stay untouched. The floor is a value-dependent post-step at apply time: clip at `floor`, then multiplicatively rescale each source-cell block's deviations-from-floor so the block mean equals the source value (closed-form, one pass). Spec: `docs/superpowers/specs/2026-06-06-floor-mean-preserving-downscale-design.md`.

**Tech Stack:** Python ≥3.12, NumPy, scipy.sparse, xarray, pytest (+hypothesis, already a dev dep).

**Pre-existing bug folded in as Task 1:** `resample_grid` on a descending-latitude grid currently returns wrong (latitude-mirrored) values — the `Resampler` is built on the raw descending coords while `_apply_matrix_da` sorts the data ascending before the matmul. Verified empirically (max diff ≈1.45 on a random 4×5 field vs the ascending twin). The floor feature's `parent_flat` indexing depends on this being consistent, so it's fixed first.

**Conventions that apply to every task:**
- Run single test files as `uv run pytest -p no:cacheprovider --no-cov <path> -v` (the full suite enforces a coverage gate that partial runs would fail).
- pytest runs with `filterwarnings = ["error", ...]` — any NumPy RuntimeWarning fails the test. Division-by-zero/invalid ops in `floor_blocks` must sit inside `np.errstate(...)`.
- Before each commit: `uv run ruff format . && uv run ruff check .`
- Every commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Fix descending-latitude bug in `resample_grid`

The library convention (CLAUDE.md) is that a descending grid and its flipped twin produce the same results. `resample_grid` violates it today. Fix: canonicalise the source latitudes ascending before building the `Resampler`, matching what `_apply_matrix_da` does to the data.

**Files:**
- Modify: `src/geohalo/api.py:112-124` (`resample_grid`)
- Test: `tests/api/test_resample.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_resample.py`:

```python
def test_resample_grid_descending_lats_match_ascending() -> None:
    # CLAUDE.md convention: a descending grid and its flipped twin give the same results.
    rng = np.random.default_rng(7)
    vals = rng.random((4, 5))
    lats = np.linspace(0.0, 3.0, 4)
    lons = np.linspace(0.0, 4.0, 5)
    da = _da(vals, lats, lons)
    out_asc = resample_grid(da, target_resolution=0.5, iterations=3)
    out_desc = resample_grid(da.sortby("latitude", ascending=False), target_resolution=0.5, iterations=3)
    np.testing.assert_allclose(out_desc.sortby("latitude").values, out_asc.values, atol=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample.py::test_resample_grid_descending_lats_match_ascending -v`
Expected: FAIL — `assert_allclose` mismatch (values differ by ~O(1)).

- [ ] **Step 3: Fix `resample_grid`**

In `src/geohalo/api.py`, change the first line of `resample_grid`'s body from:

```python
    src_lat = source[lat_dim].to_numpy()
```

to:

```python
    src_lat, _ = ensure_ascending_lats(source[lat_dim].to_numpy())
```

(`ensure_ascending_lats` is already imported at `src/geohalo/api.py:13`.) The `Resampler` is then built on ascending lats, consistent with `_apply_matrix_da`'s `sortby`, and `target_coords_from_resolution` already produces ascending targets.

- [ ] **Step 4: Run the test file to verify it passes**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample.py -v`
Expected: all PASS (new test plus the four existing ones).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: PASS, coverage gate ≥85 % satisfied.

- [ ] **Step 6: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/geohalo/api.py tests/api/test_resample.py
git commit -m "fix(api): canonicalise latitudes ascending in resample_grid

A descending-lat grid built the Resampler on descending coords while
_apply_matrix_da sorted the data ascending, latitude-mirroring the output.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Extract `parent_flat_2d` into `geometry.py`

The floor post-step needs the nearest-parent index per target cell — the exact array `_build_factors` already computes inline (`src/geohalo/resampler.py:78-80`). Extract it as a pure helper (DRY) so both call sites share it.

**Files:**
- Modify: `src/geohalo/geometry.py` (add function after `nearest_index`)
- Modify: `src/geohalo/resampler.py:9,78-80` (use the helper)
- Test: `tests/geometry/test_resample_blocks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_resample_blocks.py`:

```python
def test_parent_flat_2d_refine() -> None:
    from geohalo.geometry import parent_flat_2d

    s_lat = np.array([0.0, 1.0])
    s_lon = np.array([0.0, 1.0])
    t_lat = np.array([0.0, 0.4, 0.6, 1.0])
    t_lon = np.array([0.0, 0.4, 0.6, 1.0])
    # nearest parent per axis is [0, 0, 1, 1]; flat = lat_parent * n_s_lon + lon_parent
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ]
    ).ravel()
    np.testing.assert_array_equal(parent_flat_2d(s_lat, s_lon, t_lat, t_lon), expected)
```

(Note: `nearest_index` ties go to the lower index, so 0.4 → 0 and 0.6 → 1; no ties here.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/geometry/test_resample_blocks.py::test_parent_flat_2d_refine -v`
Expected: FAIL — `ImportError: cannot import name 'parent_flat_2d'`.

- [ ] **Step 3: Implement the helper**

In `src/geohalo/geometry.py`, after `nearest_index`, add:

```python
def parent_flat_2d(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Flat source-cell index of the nearest parent for each target cell.

    Shape ``(n_target_lat * n_target_lon,)``; flat index is
    ``lat_parent * n_source_lon + lon_parent``, matching the row-major
    flattening used everywhere else.
    """
    parent_lat = nearest_index(source_lat, target_lat)
    parent_lon = nearest_index(source_lon, target_lon)
    return (parent_lat[:, None] * source_lon.size + parent_lon[None, :]).ravel()
```

- [ ] **Step 4: Use it in `_build_factors`**

In `src/geohalo/resampler.py`, change the import (line 9) to:

```python
from geohalo.geometry import bilinear_matrix_1d, parent_flat_2d
```

and in `_build_factors`, replace:

```python
    parent_lat = nearest_index(source_lat, target_lat)
    parent_lon = nearest_index(source_lon, target_lon)
    parent_flat = (parent_lat[:, None] * n_s_lon + parent_lon[None, :]).ravel()
```

with:

```python
    parent_flat = parent_flat_2d(source_lat, source_lon, target_lat, target_lon)
```

(`n_s_lon` is still used two lines above for `n_s`; keep it.)

- [ ] **Step 5: Run geometry + resampler + api tests**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/geometry/ tests/api/test_resample.py -v`
Expected: all PASS (the refactor must not change any resampler output).

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add src/geohalo/geometry.py src/geohalo/resampler.py tests/geometry/test_resample_blocks.py
git commit -m "refactor(geometry): extract parent_flat_2d from _build_factors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `floor_blocks` — the clip-and-rescale post-step

Pure NumPy/Scipy function in `geometry.py` (no operator state, consistent with that module's charter). Implements the spec's four block rules.

**Files:**
- Modify: `src/geohalo/geometry.py` (add function after `parent_flat_2d`)
- Test: `tests/geometry/test_floor_blocks.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_floor_blocks.py`:

```python
import numpy as np

from geohalo.geometry import floor_blocks


def test_clip_and_rescale_restores_block_means() -> None:
    parent_flat = np.array([0, 0, 0, 1, 1, 1])
    source = np.array([[1.0, 2.0]])
    # block 0 has a negative child; block 1 is clean
    resampled = np.array([[-1.0, 1.0, 3.0, 2.0, 2.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert out.min() >= 0.0
    np.testing.assert_allclose(out[0, :3].mean(), 1.0, atol=1e-12)
    np.testing.assert_allclose(out[0, 3:], [2.0, 2.0, 2.0], atol=1e-12)


def test_noop_when_above_floor_and_mean_matching() -> None:
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[2.0, 4.0]])
    resampled = np.array([[1.5, 2.5, 3.0, 5.0]])  # block means already 2.0 and 4.0
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_allclose(out, resampled, atol=1e-12)


def test_source_below_floor_fills_block_with_floor() -> None:
    # rule 2: mean preservation and the floor are mutually impossible -> floor wins
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[-0.5, 2.0]])
    resampled = np.array([[0.2, -1.2, 2.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_array_equal(out[0, :2], [0.0, 0.0])
    np.testing.assert_allclose(out[0, 2:].mean(), 2.0, atol=1e-12)


def test_all_children_clipped_fills_block_with_parent() -> None:
    # rule 3: the rescale is 0/0 -> constant parent fill keeps mean and floor
    parent_flat = np.array([0, 0])
    source = np.array([[1.0]])
    resampled = np.array([[-2.0, -3.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_array_equal(out, [[1.0, 1.0]])


def test_nan_block_stays_nan_neighbours_unaffected() -> None:
    # rule 1: one NaN child blanks its whole block (same footprint as the
    # linear path, where P(x - A@y) broadcasts the NaN residual block-wide)
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[1.0, 3.0]])
    resampled = np.array([[np.nan, 0.5, -1.0, 5.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert np.isnan(out[0, :2]).all()
    assert out[0, 2:].min() >= 0.0
    np.testing.assert_allclose(out[0, 2:].mean(), 3.0, atol=1e-12)


def test_nan_source_keeps_block_nan() -> None:
    parent_flat = np.array([0, 0])
    source = np.array([[np.nan]])
    resampled = np.array([[1.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert np.isnan(out).all()


def test_nonzero_floor() -> None:
    parent_flat = np.array([0, 0])
    source = np.array([[2.0]])
    resampled = np.array([[0.0, 3.0]])  # 0.0 is below floor=1.0
    out = floor_blocks(resampled, source, parent_flat, 1.0)
    assert out.min() >= 1.0
    np.testing.assert_allclose(out.mean(), 2.0, atol=1e-12)


def test_batched_rows_independent() -> None:
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    resampled = np.array([[-1.0, 1.0, 2.0, 2.0], [3.0, 3.0, -4.0, 4.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    row0 = floor_blocks(resampled[:1], source[:1], parent_flat, 0.0)
    row1 = floor_blocks(resampled[1:], source[1:], parent_flat, 0.0)
    np.testing.assert_allclose(out, np.vstack([row0, row1]), atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/geometry/test_floor_blocks.py -v`
Expected: all FAIL — `ImportError: cannot import name 'floor_blocks'`.

- [ ] **Step 3: Implement `floor_blocks`**

In `src/geohalo/geometry.py`, after `parent_flat_2d`, add:

```python
def floor_blocks(
    resampled: np.ndarray,
    source_flat: np.ndarray,
    parent_flat: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Clip a resampled batch at `floor`, restoring each source-cell block's mean.

    `resampled` is ``(batch, n_target)``, `source_flat` is ``(batch, n_source)``,
    `parent_flat` maps each target cell to its nearest source cell (see
    :func:`parent_flat_2d`). Each block's deviations-from-floor are rescaled
    multiplicatively, so the (uniform) child mean equals the source value
    exactly while staying ``>= floor``. Rules, in precedence order:

    1. blocks that contain NaN stay NaN — the NaN footprint is unchanged;
    2. a source value below `floor` fills its block with `floor` (mean
       knowingly broken only where the input already violated the floor);
    3. a block whose children all clipped to `floor` is filled with the
       parent value;
    4. otherwise: ``out = floor + (clip(y) - floor) * (x - floor) / mean(clip(y) - floor)``.
    """
    n_source = source_flat.shape[-1]
    n_target = parent_flat.size
    counts = np.bincount(parent_flat, minlength=n_source)
    adj = sp.csr_matrix(
        (np.ones(n_target), (parent_flat, np.arange(n_target))),
        shape=(n_source, n_target),
    )
    deviation = np.maximum(resampled, floor) - floor  # >= 0; NaN propagates
    block_sum = np.asarray(deviation @ adj.T)  # (batch, n_source); NaN children poison their block
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dev = block_sum / np.maximum(counts, 1)
        scale = (source_flat - floor) / mean_dev
        out = floor + deviation * scale[..., parent_flat]
    child_src = source_flat[..., parent_flat]
    # rule 3: every child clipped (scale was 0/0 or inf) -> constant parent fill
    out = np.where((mean_dev == 0.0)[..., parent_flat], child_src, out)
    # rule 2: source already violates the floor -> constant floor fill
    return np.where(child_src < floor, floor, out)
```

Why the rules fall out correctly:
- NaN anywhere in a block → `block_sum` NaN → `scale` NaN → block NaN; `mean_dev == 0.0` is False for NaN and `child_src < floor` is False for finite source, so neither `np.where` overwrites it (rule 1).
- `mean_dev == 0` with parent above floor → `scale` is inf, `deviation * inf` is 0·inf = NaN — overwritten by the parent fill (rule 3). Parent exactly at floor → fill is the floor, consistent.
- `scipy.sparse as sp` is already imported in `geometry.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/geometry/test_floor_blocks.py -v`
Expected: 8 PASS, no warnings (the `errstate` block is what keeps `filterwarnings=error` happy).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/geohalo/geometry.py tests/geometry/test_floor_blocks.py
git commit -m "feat(geometry): floor_blocks clip-and-rescale post-step

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire `floor` through the DataArray path

`floor` lands on `_apply_matrix_da` (between matmul and reshape — both flat buffers are in hand there), `resample_grid_with_matrix`, and `resample_grid`. Mapping input on a DataArray raises. No digest, serialization, or `PAYLOAD_VERSION` change anywhere.

**Files:**
- Modify: `src/geohalo/api.py` (imports; `_apply_matrix_da`; `resample_grid_with_matrix`; `resample_grid`)
- Test: `tests/api/test_resample_floor.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_resample_floor.py`:

```python
import numpy as np
import pytest
import xarray as xr

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
    da, lats, lons = _sharp_field()
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
    with pytest.raises(ValueError, match="Dataset"):
        resample_grid(da, target_resolution=0.25, floor={"tp": 0.0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample_floor.py -v`
Expected: `test_unfloored_baseline_goes_negative` PASSES (it exercises current behaviour); every floor test FAILS with `TypeError: ... unexpected keyword argument 'floor'`.

- [ ] **Step 3: Implement the wiring**

In `src/geohalo/api.py`:

1. Extend the imports:

```python
from collections.abc import Callable, Hashable, Mapping
```

and add `floor_blocks, parent_flat_2d` to the `geohalo.geometry` import:

```python
from geohalo.geometry import (
    ensure_ascending_lats,
    floor_blocks,
    parent_flat_2d,
    same_grid,
    target_coords_from_resolution,
)
```

2. `_apply_matrix_da` — add two trailing keyword-only params and the post-step:

```python
def _apply_matrix_da(
    da: xr.DataArray,
    matrix: sp.csr_matrix,
    lat_dim: str,
    lon_dim: str,
    out_lat: np.ndarray,
    out_lon: np.ndarray,
    *,
    floor: float | None = None,
    parent_flat: np.ndarray | None = None,
) -> xr.DataArray:
```

and after the existing `out_flat = np.asarray(flat @ matrix.T)` line insert:

```python
    if floor is not None and parent_flat is not None:
        # value-dependent post-step: clip at `floor`, restore each source-cell
        # block's mean (see geometry.floor_blocks). T itself stays linear/cached.
        out_flat = floor_blocks(out_flat, flat, parent_flat, floor)
```

3. `resample_grid_with_matrix` — full replacement:

```python
def resample_grid_with_matrix[T: xr.DataArray | xr.Dataset](
    source: T,
    resampler: Resampler,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    floor: float | Mapping[Hashable, float] | None = None,
) -> T:
    """Resample with a prebuilt :class:`Resampler`.

    `floor` clips the output at a lower bound while preserving each source
    cell's child mean exactly (e.g. ``floor=0.0`` for precipitation). For a
    Dataset it may be a mapping ``{var_name: floor}`` to bound only the named
    spatial variables.
    """
    if isinstance(source, xr.Dataset):
        floors = _floor_by_var(floor, source, lat_dim, lon_dim)
        return _map_spatial_vars(
            source,
            lambda da: resample_grid_with_matrix(
                da, resampler, lat_dim=lat_dim, lon_dim=lon_dim, floor=floors.get(da.name),
            ),
            lat_dim,
            lon_dim,
        )
    if isinstance(floor, Mapping):
        raise ValueError("a floor mapping is Dataset-only; pass a float for DataArray input")
    parent = None
    if floor is not None:
        parent = parent_flat_2d(
            resampler.source_lat, resampler.source_lon, resampler.target_lat, resampler.target_lon,
        )
    return _apply_matrix_da(
        source, resampler.transform_matrix, lat_dim, lon_dim,
        resampler.target_lat, resampler.target_lon,
        floor=floor, parent_flat=parent,
    )
```

4. `_floor_by_var` — **stub for this task** so the DataArray path compiles; Task 5 gives it real mapping validation. Place it directly above `resample_grid_with_matrix`:

```python
def _floor_by_var(
    floor: float | Mapping[Hashable, float] | None,
    ds: xr.Dataset,
    lat_dim: str,
    lon_dim: str,
) -> dict[Hashable, float]:
    """Resolve `floor` to a per-spatial-var dict; floats apply to every spatial var."""
    if floor is None:
        return {}
    spatial = [n for n, v in ds.data_vars.items() if lat_dim in v.dims and lon_dim in v.dims]
    if isinstance(floor, Mapping):
        unknown = set(floor) - set(spatial)
        if unknown:
            raise ValueError(
                f"floor names variables that are not spatial data vars: {sorted(map(str, unknown))}",
            )
        return dict(floor)
    return {n: float(floor) for n in spatial}
```

5. `resample_grid` — add the param and pass it through:

```python
def resample_grid[T: xr.DataArray | xr.Dataset](
    source: T,
    target_resolution: float,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    iterations: int = 1,
    floor: float | Mapping[Hashable, float] | None = None,
) -> T:
    """Mean-preserving resample onto a `target_resolution` grid.

    `floor` clips the output at a lower bound while preserving each source
    cell's child mean exactly (e.g. ``floor=0.0`` for precipitation). For a
    Dataset it may be a mapping ``{var_name: floor}`` to bound only the named
    spatial variables.
    """
    src_lat, _ = ensure_ascending_lats(source[lat_dim].to_numpy())
    src_lon = source[lon_dim].to_numpy()
    t_lat, t_lon = target_coords_from_resolution(src_lat, src_lon, target_resolution)
    resampler = Resampler.compute(src_lat, src_lon, t_lat, t_lon, iterations=iterations)
    return resample_grid_with_matrix(source, resampler, lat_dim=lat_dim, lon_dim=lon_dim, floor=floor)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample_floor.py tests/api/test_resample.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add src/geohalo/api.py tests/api/test_resample_floor.py
git commit -m "feat(api): floor= on resample_grid for non-negative downscaling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Dataset semantics for `floor`

`_floor_by_var` exists from Task 4; this task pins its behaviour with tests (float → all spatial vars; mapping → named vars only, others pass through linear; unknown names raise).

**Files:**
- Modify: `src/geohalo/api.py` (only if a test exposes a gap)
- Test: `tests/api/test_resample_floor.py`

- [ ] **Step 1: Write the tests**

Append to `tests/api/test_resample_floor.py`:

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample_floor.py -v`
Expected: all PASS with the Task 4 implementation. If any fail, the gap is in `_floor_by_var` — fix it there (its contract is exactly these four tests).

- [ ] **Step 3: Run the full suite and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add src/geohalo/api.py tests/api/test_resample_floor.py
git commit -m "test(api): pin per-variable floor semantics for Dataset inputs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cross-cutting integration tests (descending lats, batch dims, NaN, property test)

**Files:**
- Test: `tests/api/test_resample_floor.py`

- [ ] **Step 1: Write the tests**

Append to `tests/api/test_resample_floor.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


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
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest -p no:cacheprovider --no-cov tests/api/test_resample_floor.py -v`
Expected: all PASS. Watch for two failure modes worth knowing in advance:
- The property test repeatedly building resamplers is the slow part; `max_examples=25` keeps it a few seconds. Do not add fixtures to the `@given` test (hypothesis + function-scoped fixtures triggers a health-check error).
- If the descending test fails, the bug is in `parent_flat` ordering vs the sorted data — re-read Task 1 and the spec's "implementation caution" before touching anything.

- [ ] **Step 3: Run the full suite and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add tests/api/test_resample_floor.py
git commit -m "test(api): floor integration and hypothesis property tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/concepts/downscaling.md` (new section before "Two forms of the resampler")
- Modify: `docs/guides/resampling.md` (mention `floor`)

- [ ] **Step 1: Add the concepts section**

In `docs/concepts/downscaling.md`, insert a new section immediately **before** the `## Two forms of the resampler` heading:

```markdown
## Non-negative variables (`floor=`)

The refinement series and the per-block correction are *signed*: near a sharp
gradient (a wet cell beside dry ones) the smooth surface can overshoot below
zero, and the block-constant correction \(\mathbf{P}(x - \mathbf{A}y)\) can push
an entire dry block negative — even though bilinear interpolation alone never
would. For precipitation, wind speed, or concentrations that is unphysical.

`floor=` clips the output at a lower bound **without giving up mean
preservation**. After the linear transform, each source-cell block is clipped
and its deviations-from-floor rescaled in closed form:

\[
y'' = f + \bigl(\max(y, f) - f\bigr)\cdot
      \frac{x_\text{parent} - f}{\overline{\max(y, f) - f}}
\]

so every child stays \(\ge f\) *and* the block's child mean equals the source
value exactly — one pass, no iteration.

```python
fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3, floor=0.0)
```

Details that matter:

- The transform matrix \(\mathbf{T}\) stays linear and cacheable — the floor is
  a value-dependent post-step at apply time, so digests and caches are
  untouched.
- A source cell already below the floor (e.g. a tiny negative precipitation
  artifact in the input) gets its block filled with the floor: the bound is
  guaranteed everywhere, and the mean is knowingly broken only where the input
  itself violated it.
- Blocks whose children all clip are filled with the parent value; NaN blocks
  stay NaN.
- For a `Dataset`, pass a mapping to bound only some variables:
  `floor={"tp": 0.0}`.
- `reduce(..., target_resolution=...)` does **not** take `floor` — the fused
  operator never materialises the fine field, and a nonlinear clip cannot pass
  through the fusion. When per-polygon values must respect the bound, do the
  explicit two-step: `resample_grid(..., floor=0.0)` then `reduce`.
```

- [ ] **Step 2: Mention `floor` in the resampling guide**

In `docs/guides/resampling.md`, insert a new paragraph directly after the line
`` `resample_grid` accepts an `xr.DataArray` or an `xr.Dataset` (every spatial data variable ``
`` is resampled, the rest pass through) and preserves all non-spatial dims. `` (lines 20–21, end of the "One call" section):

```markdown
For non-negative variables such as precipitation, add `floor=0.0` — the output
is clipped at the bound while each source cell's child mean is still preserved
exactly. See [Mean-preserving downscaling](../concepts/downscaling.md#non-negative-variables-floor)
for how the clip-and-rescale works.
```

Note on spec coverage: the spec asks that "the `reduce` docs gain one line pointing
floor-needing users at the two-step". `reduce` has no docstring today, so that pointer
lives in the last bullet of the new concepts section above (and in the guide paragraph) —
do not add a one-line docstring to `reduce` just for this.

- [ ] **Step 3: Verify the docs build**

Run: `uv run --group docs mkdocs build --strict 2>&1 | tail -5`
Expected: `... Documentation built ...` with no warnings (`--strict` turns warnings into errors; a broken anchor in the cross-link would fail here).

- [ ] **Step 4: Run the full suite one last time and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add docs/concepts/downscaling.md docs/guides/resampling.md
git commit -m "docs: floored mean-preserving downscaling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
