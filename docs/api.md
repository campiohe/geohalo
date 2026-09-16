# API reference

High-level entry points are exported from the top-level `geohalo` package. Each
workflow comes in a **convenience** form (builds the operator for you) and a
**precomputed** form (you pass a cached object).

| Convenience            | Precomputed                  | Builds                                       |
| ---------------------- | ---------------------------- | -------------------------------------------- |
| `reduce`               | `reduce_with_stencil` / `reduce_with_operator` | [`Stencil`](concepts/stencil.md) / [`ReduceOperator`](concepts/reduce-operator.md) |
| `resample_grid`        | `resample_grid_with_matrix`  | [`Resampler`](guides/resampling.md)       |
| `aggregate_bias`       | `aggregate_bias_with_tree`   | [`BiasTree`](concepts/bias-tree.md)          |

---

## Reduction

::: geohalo.reduce

::: geohalo.reduce_with_stencil

::: geohalo.reduce_with_operator

::: geohalo.reduce_with_restricted_operator

---

## Resampling

::: geohalo.resample_grid

::: geohalo.resample_grid_with_matrix

---

## Hierarchical rollups

::: geohalo.aggregate_bias

::: geohalo.aggregate_bias_with_tree

---

## Precomputed operators

All five classes support `obj.to_npz() -> bytes` and
`Class.from_npz(blob: bytes) -> Class`. These versioned, non-pickle artifacts can
be stored independently of the built-in caches. See
[portable operator artifacts](concepts/serialization.md) for supported keys,
the array schema, validation, and migration from older caches.

::: geohalo.Stencil

::: geohalo.ReduceOperator

::: geohalo.RestrictedOperator

::: geohalo.Resampler

::: geohalo.BiasTree

---

## Caching

::: geohalo.LocalCache

::: geohalo.RedisCache

---

## Geometry areas

These helpers live in `geohalo.geometry`. See
[latitude correction](concepts/latitude-correction.md#polygon-areas) for the
coordinate convention and the distinction between polygon areas and stencil weights.

::: geohalo.geometry.polygon_areas

::: geohalo.geometry.cell_areas

---

## Exceptions

::: geohalo.EmptyOverlapError
