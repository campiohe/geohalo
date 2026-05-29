# Mean-preserving resampling

`geohalo` resamples a source grid to any target grid (refine, coarsen, or
shift) with a single value-independent sparse matrix `T`
(`target_flat = T @ source_flat`). Because the matrix doesn't depend on the
data, it is built once per grid pair and reused — and cached via
`LocalCache.get_or_compute_resampler`.

## Construction

From the source and target coordinate arrays, build three value-independent
matrices:

- `B` — bilinear interpolation (target ← source), separable as
  `kron(B_lat, B_lon)` where each 1-D factor maps source centres to target
  centres with clamped edges.
- `π` — a nearest-cell assignment of each target cell to a source cell. From
  it:
  - `P` — source → target broadcast (`P[t, π(t)] = 1`).
  - `A` — source ← target mean (`A[s, t] = 1/n_s` when `π(t) = s`, where
    `n_s` is the number of target cells assigned to source cell `s`; a source
    cell with no assigned target has an all-zero row).

The iterative interpolate-and-correct process is linear in the source values
`x`:

```
y₀ = B·x
smoothing (× (iterations − 1)):  y ← y + B·(x − A·y)
final hard correction:           y ← y + P·(x − A·y)
```

Every step is linear, so the whole thing collapses to a matrix:

```
G    = I − B·A
y_op = (Σ_{j=0}^{iterations−1} Gʲ) · B
T    = y_op + P·(I − A·y_op)
```

Since `A·P = diag(n_s > 0)`, the hard correction gives `A·T = I` on every
source cell with at least one assigned target cell → **exact
mean-preservation**: averaging a source cell's target children recovers the
source value. This always holds when refining; when coarsening, source cells
with no assigned target can't be preserved (geometrically unavoidable).

With `iterations = 1` this reduces exactly to the classic downscale operator
`M = B + P − P·A·B`.

## Cost

`y_op = (Σⱼ Gʲ)·B` is never built by forming `G = I − B·A` (an `n_target ×
n_target` operator that densifies under repeated products). Instead the series
is accumulated by applying `G` to `B` on the right:

```
acc = term = B
repeat (iterations − 1):  term ← term − B·(A·term);  acc ← acc + term
y_op = acc
```

Every intermediate stays `n_target × n_source`, so the dense `n_target ×
n_target` matrix is never materialised. The build is sub-second even for large
refinements; cost still grows with `iterations` because the transform fills in
as its reach expands, but only the (sparse) result is held. The default
`iterations = 1` is the cheapest case. See the `Resampler.compute` rows in the
README benchmark table.

## Fusing into the reduce path

When the resampled grid is only an intermediate on the way to per-polygon
values, you never need `T` at all. The reduce path composes the resample with
the stencil's occupancy matrix `W` into a single operator `W·T` that acts on
the source grid, building it thinly without materialising `T` or the fine
field. On a fine target grid this is the difference between a multi-gigabyte
(or un-buildable) `T` and a sub-megabyte fused operator. See
[`reduce-operator.md`](reduce-operator.md).
