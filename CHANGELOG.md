# Changelog

## Unreleased

- Add opt-in `partial_cell_weighting="exact"` to `Stencil.compute`, `reduce`,
  and LocalCache/RedisCache stencil builders. Boundary cells use spherical
  polygon-cell intersection areas with straight lon/lat edges, including holes
  and multipart polygons. Requires `spherical_correction=True`. The default
  `"approximate"` weights, results, and cache keys remain unchanged. NPZ artifacts
  retain the option; older stencils load as approximate. Fused and restricted
  reductions inherit the chosen weights and distinct cache identities.

## 2.0.0 — 2026-09-16

### Migration from 1.2.0

- **Polygon row order:** stencil/operator rows and reduction results now follow
  caller order, including cache hits. Remove permutations that undid the old
  `repr(key)` sorting, or sort the input GeoSeries first to retain sorted output.
  `BiasTree` ordering and label-based alignment are unchanged.
- **Cache format:** local `.pkl` entries and legacy Redis prefixes are ignored
  and rebuilt using versioned, non-pickle NPZ artifacts. Old entries are neither
  loaded nor deleted automatically; expect a one-time rebuild. Custom key types
  that cannot be encoded safely now fail explicitly. See
  [portable artifacts and migration](https://campiohe.github.io/geohalo/concepts/serialization/).
- Existing mean-preserving resampling, float64 coefficients, dtype promotion,
  and NaN propagation remain defaults. Conservative regridding, periodic
  longitude, float32 coefficients, dtype preservation, and NaN skipping are opt-in.

### Changes

- Add `geohalo.geometry.polygon_areas` for zone areas in m² on the same sphere
  as `cell_areas` ([#21](https://github.com/campiohe/geohalo/issues/21)). Integrates
  straight lon/lat edges analytically, including sloped edges, holes, multipart
  geometries, and full-globe rectangles. Preserves input order, returns NaN for
  missing geometries, validates coordinates/topology/declared CRS, and leaves
  longitudes unwrapped. Existing cell areas, stencil weights, and reductions
  are unchanged; partial-cell stencil areas remain an approximation.

- Add byte-based `to_npz()` / `from_npz()` to `Stencil`, `ReduceOperator`,
  `Resampler`, `RestrictedOperator`, and `BiasTree`
  ([#20](https://github.com/campiohe/geohalo/issues/20)). Versioned NumPy arrays
  and typed JSON metadata preserve sparse buffers, grids, normalizers, read
  plans, and pandas keys without executable object serialization. Both
  resampler representations are supported. Loads disable pickle and validate
  structure; cache hits also check the full input digest. **Cache migration:**
  LocalCache now uses `.npz` files and Redis uses `:npz:v1:` prefixes. Legacy
  entries are ignored and rebuilt, never automatically loaded or deleted.
  Build-input digests remain unchanged. Unsupported custom key types fail explicitly.

- Add `method="conservative"` to `Resampler.compute`, `resample_grid`, and
  local/Redis resampler caches ([#15](https://github.com/campiohe/geohalo/issues/15)).
  Two sparse 1-D overlap factors use sin(latitude) and longitude widths, applied
  per slice without a grid-to-grid Kronecker matrix. Full target-area
  normalization is the default; `normalization="covered"` and opt-in
  `skipna=True` support covered/valid-area means. Unmapped cells return NaN.
  Supports explicit cell bounds, singleton axes with bounds, descending axes,
  polar clipping, and periodic longitude. Default mean-preserving calls and
  their input digests are unchanged; conservative NPZ artifacts store the two
  factors. Fused polygon reduction is unchanged.

- Add opt-in longitude periodicity with `period=360`
  ([#22](https://github.com/campiohe/geohalo/issues/22)). Linear interpolation
  and nearest-parent assignment wrap together in materialized/factored
  resamplers, fused reductions, and local/Redis caches. With `period` set,
  resolution-based helpers generate a full longitude cycle without a repeated
  endpoint; explicit target arrays retain their coordinates and order.
  Nonperiodic defaults and cache keys are unchanged. Restricted plans inherit
  seam-crossing coefficients and read only contributing chunks. Periodic
  interpolation does not wrap polygon geometries.

- Add opt-in `dtype=np.float32` to `Stencil.compute`, `ReduceOperator.compute`,
  and their LocalCache/RedisCache methods, plus `preserve_dtype=True` on all
  polygon reducers, `ReduceOperator.apply_grid`, and `RestrictedOperator.apply`
  ([#19](https://github.com/campiohe/geohalo/issues/19)). Float32 coefficients
  reduce storage while row normalizers remain float64. Floating-point results
  can retain each input variable's dtype; integer means remain floating-point.
  Existing float64 defaults, input digests, and NaN semantics are
  unchanged. Float32 operators use separate cache keys; restricted plans inherit
  coefficient dtype. Reduced precision is opt-in, not an accuracy guarantee.

- Preserve the caller's polygon order in `Stencil.compute`, reduction matrix
  rows, and NumPy/xarray reduction outputs
  ([#18](https://github.com/campiohe/geohalo/issues/18)). Local and Redis caches
  store canonical rows and restore the requested order, keeping input digests
  order-invariant for unique keys. **Compatibility:** outputs no longer sort polygon
  keys; remove any positional permutation previously used to undo that sort.
  To retain the previous order, sort the input GeoSeries by `repr(key)` first.
  `BiasTree` node order is unchanged; xarray aggregation aligns leaves by key.

- Add opt-in `skipna=True` to `reduce_with_operator`,
  `reduce_with_restricted_operator`, `RestrictedOperator.apply`, and
  `ReduceOperator.apply_grid` ([#16](https://github.com/campiohe/geohalo/issues/16)).
  Means renormalize over surviving source-cell weights and return NaN for
  nonpositive denominators; sums omit missing contributions (zero if all are
  missing). Masks are bounded per slice, and restricted reads do not change.
  Default NaN propagation and the stencil path's
  resample-then-mask semantics are unchanged.

- Add `RestrictedOperator.gather` and `RestrictedOperator.apply` for reducing
  NumPy arrays supplied by caller-owned readers
  ([#17](https://github.com/campiohe/geohalo/issues/17)). Window iterators support
  one-window-at-a-time gathering; the xarray adapter shares the same array
  methods. Mean/sum arithmetic, dtype promotion, and NaN propagation are
  unchanged. No new dependencies.

## 1.2.0 — 2026-09-16

- Add `RestrictedOperator`, `reduce_with_restricted_operator`, and local/Redis
  read-plan caching for chunk-aware reduction of clean lazy grids
  ([#6](https://github.com/campiohe/geohalo/issues/6)). Plans infer Dask/backend
  chunk metadata or accept explicit irregular chunks, preserve stored latitude
  order, and read disjoint windows containing only contributing spatial chunks.
  Batch chunks are processed separately; results are eager. Existing reduction
  entry points and missing-data semantics are unchanged.
- Accelerate `BiasTree.compute` with depth-bounded sparse products instead of
  composing internal nodes one sparse row at a time
  ([#8](https://github.com/campiohe/geohalo/issues/8)). Weighted means and sums,
  node ordering, hierarchy validation, and existing cache entries are preserved.
- Accelerate stencil construction by passing vectorised WKB geometries directly
  to exactextract and reusing those bytes for the geometry digest
  ([#7](https://github.com/campiohe/geohalo/issues/7)). Coverage matrices, polygon
  key ordering, and cache digests are unchanged. Missing geometries passed to
  `Stencil.compute` now raise `ValueError` before native extraction.
- Reduce temporary memory when applying `ReduceOperator` to large grids
  ([#5](https://github.com/campiohe/geohalo/issues/5)): gather contributing cells
  one batch slice at a time and reuse the compact column mapping. Resampling
  also uses per-slice or bounded-batch products. Both paths handle descending
  latitudes without sorting the full input and preserve matrix precision.
- Fix north/south reversal when resampling grids with descending source latitudes
  ([#4](https://github.com/campiohe/geohalo/issues/4)). `Resampler.compute` and
  `FactoredResampler.compute` now store source latitudes ascending and build their
  matrices in that order. The xarray helpers accept either latitude orientation.
- Share resampler cache entries between ascending and descending versions of the
  same source grid. Existing ascending-source entries remain valid; old
  descending-source entries are no longer used.
- Reject mismatched source coordinates in `resample_grid_with_matrix` with a
  `ValueError`.

Compatibility: code that directly multiplies `Resampler.transform_matrix` or calls
`FactoredResampler.apply_flat` must now flatten source values in ascending latitude
order, with longitude in `source_lon` order. Target coordinate order is unchanged.
