# Hierarchical rollups

Once you have one value per leaf polygon (municipalities, counties, basins), you often
want them rolled up to a parent grouping (states, regions, countries). geohalo treats
this the same way it treats everything else: **as a linear operator** you precompute once
and apply with a matmul.

## The rollup matrix

Let \(\mathbf{a}\) be the leaf values. A rollup to all nodes (leaves *and* internal
parents) is

\[
\mathbf{r} \;=\; \mathbf{R}\,\mathbf{a},
\qquad
\mathbf{R} \in \mathbb{R}^{N_\text{nodes} \times N_\text{leaves}}
\]

Each row of \(\mathbf{R}\) expresses a node as a combination of the **leaves that
transitively contribute to it**:

- a **leaf** row is a single 1 (it is itself);
- an **internal** row is the normalised (for `how="mean"`) or raw (for `how="sum"`)
  weighted combination of its children's rows, composed recursively up the tree.

```mermaid
flowchart TD
    BR["BR<br/>(country)"]
    SP["SP<br/>(state)"]
    RJ["RJ<br/>(state)"]
    a["muni_a"] --> SP
    b["muni_b"] --> SP
    c["muni_c"] --> RJ
    SP --> BR
    RJ --> BR
```

`BiasTree.compute` builds a sparse adjacency matrix \(\mathbf{A}\), with each
parent–child entry set to `w/total` (mean) or `w` (sum), and a rectangular identity
\(\mathbf{L}\) that selects the leaves. Starting from \(\mathbf{R}_0=\mathbf{L}\),
it computes

\[
\mathbf{R}_{k+1} = \mathbf{L} + \mathbf{A}\,\mathbf{R}_k.
\]

Each sparse product propagates leaf contributions one edge toward the roots.
After as many products as the maximum leaf-to-root depth, the matrix is complete,
including leaves attached at different depths and disconnected trees (a forest).
The iteration count comes from the validated hierarchy, not a floating-point
convergence test. This avoids repeatedly copying and adding individual sparse
rows. Build cost still depends on tree depth and the number of leaf–ancestor
coefficients in the result; very deep chains need more products than shallow trees.

Node ordering is unchanged: leaves sorted by `repr`, then internal nodes by
depth and `repr`. Cache input digests are unchanged; payloads now use
[portable NPZ serialization](serialization.md).

## Usage

```python
import pandas as pd
import geohalo as ghl

edges = pd.DataFrame(
    {"parent": [("BR", "SP"), ("BR", "SP"), ("BR", "RJ")]},
    index=pd.Index(
        [("BR", "SP", "muni_a"), ("BR", "SP", "muni_b"), ("BR", "RJ", "muni_c")],
        name="child",
    ),
)

rolled = ghl.aggregate_bias(leaf_aggregates, edges)
```

The DataFrame **index is the child**; the `parent` column is its parent. The output
carries every node — leaves *and* internal parents — on the `geom` dim, so you can read
a municipality and its state from the same array.

## Options

| Argument     | Default    | Effect                                                          |
| ------------ | ---------- | --------------------------------------------------------------- |
| `how`        | `"mean"`   | `"mean"` normalises each parent's children; `"sum"` adds them   |
| `weight_col` | `None`     | column of per-edge weights (e.g. child area or population)      |
| `parent_col` | `"parent"` | which column names the parent                                   |

```python
rolled = ghl.aggregate_bias(leaf_aggregates, edges, how="sum")
rolled = ghl.aggregate_bias(leaf_aggregates, edges, weight_col="area")   # area-weighted
```

## Rules geohalo enforces

- **Tree, not DAG.** Each child has at most one parent — `edges.index` must be unique.
  This is checked at build time.
- **No cycles.** Every node must be reachable from some leaf; a cycle (a node unreachable
  from any leaf) raises an error naming the offending nodes.
- **Positive finite weights.** Edge weights must be positive and finite.

## NaN handling

`aggregate_bias_with_tree` mirrors the [masked reduce path](masked.md). If all leaves are
valid it is one clean matmul `a @ R.T`. If some leaves are NaN:

- for `how="mean"`, NaN leaves are **dropped** and the surviving weights renormalised
  (numerator/denominator matmuls, exactly as in the reduce path);
- for `how="sum"`, NaN leaves contribute zero.

A parent whose contributing leaves are all NaN resolves to `NaN`.

## Precompute and cache

Like the stencil, the rollup matrix depends only on the **hierarchy**, not the values —
so it is built once and [cached](../guides/caching.md) via `get_or_compute_tree`. The
hot-path rollup of a batch of 50 slices over the full Brazil muni → state hierarchy
(5 571 leaves) takes a few milliseconds.

```python
import geohalo as ghl

tree = cache.get_or_compute_tree(edges)
rolled = ghl.aggregate_bias_with_tree(leaf_aggregates, tree)
```
