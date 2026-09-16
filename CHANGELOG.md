# Changelog

## Unreleased

- Add opt-in `skipna=True` to `reduce_with_operator`,
  `reduce_with_restricted_operator`, `RestrictedOperator.apply`, and
  `ReduceOperator.apply_grid` ([#16](https://github.com/campiohe/geohalo/issues/16)).
  Means renormalize over surviving source-cell weights and return NaN for
  nonpositive denominators; sums omit missing contributions (zero if all are
  missing). Masks are bounded per slice, and restricted reads do not change.
  Default NaN propagation, cache formats, and the stencil path's
  resample-then-mask semantics are unchanged.

- Add `RestrictedOperator.gather` and `RestrictedOperator.apply` for reducing
  NumPy arrays supplied by caller-owned readers
  ([#17](https://github.com/campiohe/geohalo/issues/17)). Window iterators support
  one-window-at-a-time gathering; the xarray adapter shares the same array
  methods. Mean/sum arithmetic, dtype promotion, and NaN propagation are
  unchanged. No new dependencies or cache-format changes.

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
