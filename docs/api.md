# API reference

The public surface is everything exported from the top-level `geohalo` package. Each
entry point comes in a **convenience** form (builds the operator for you) and a
**precomputed** form (you pass a cached object).

| Convenience            | Precomputed                  | Builds                                       |
| ---------------------- | ---------------------------- | -------------------------------------------- |
| `reduce`               | `reduce_with_stencil` / `reduce_with_operator` | [`Stencil`](concepts/stencil.md) / [`ReduceOperator`](concepts/reduce-operator.md) |
| `resample_grid`        | `resample_grid_with_matrix`  | [`Resampler`](concepts/downscaling.md)       |
| `aggregate_bias`       | `aggregate_bias_with_tree`   | [`BiasTree`](concepts/bias-tree.md)          |

---

## Reduction

::: geohalo.reduce

::: geohalo.reduce_with_stencil

::: geohalo.reduce_with_operator

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

::: geohalo.Stencil

::: geohalo.ReduceOperator

::: geohalo.Resampler

::: geohalo.BiasTree

---

## Caching

::: geohalo.LocalCache

::: geohalo.RedisCache

---

## Exceptions

::: geohalo.EmptyOverlapError
