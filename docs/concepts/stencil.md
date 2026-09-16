# The stencil

The `Stencil` is geohalo's concrete realisation of the operator \(\mathbf{W}\) from
[the previous page](linear-operator.md). It is a frozen dataclass wrapping a sparse
matrix and just enough metadata to apply and validate it.

## How it is built

`Stencil.compute(lats, lons, geoms)` runs a fixed pipeline:

```mermaid
flowchart TD
    A["lat/lon centres + polygons"] --> B["derive cell edges<br/>(midpoints, EPSG:4326)"]
    B --> C["exact_extract → per-cell<br/>coverage fraction ∈ [0,1]"]
    B --> D["cell_areas → R²·Δλ·(sinφ_top−sinφ_bot)"]
    C --> E["weight = coverage × area"]
    D --> E
    E --> F["scipy.sparse CSR<br/>(n_polygons × n_cells)"]
    classDef hl fill:#fef3c7,stroke:#b45309,color:#0b1220;
```

1. **Canonicalise the grid.** Latitudes are flipped to ascending and both axes are
   checked for regular spacing — `exactextract`'s raster model assumes a uniform
   EPSG:4326 grid, and an irregular axis would silently misplace coverage fractions.

2. **Encode polygons in caller order.** Encode the geometries with one vectorised
   [`shapely.to_wkb`](https://shapely.readthedocs.io/en/stable/reference/shapely.to_wkb.html)
   call. The resulting binary geometry bytes serve both extraction and hashing.
   Only hashing uses a `repr(key)` sort, keeping the canonical
   [digest](../guides/caching.md) unchanged.

3. **Exact coverage.** A `NumPyRasterSource` describes the grid's bounding box;
   [`exact_extract`](https://github.com/isciences/exactextract) returns, for each
   polygon, the list of `cell_id`s it touches and the fraction of each cell covered.
   A WKB feature source passes the binary geometries directly to exactextract,
   avoiding Python coordinate lists and GeoJSON encoding/parsing.
   This is the unbiased boundary treatment — see
   [why exact fractional coverage](exact-coverage.md).

4. **Weight by area.** Each coverage fraction is multiplied by the cell's true
   spherical area, so a half-covered equatorial cell outweighs a half-covered polar
   one. See [latitude correction](latitude-correction.md).

5. **Assemble CSR.** The `(polygon, cell, weight)` triples become a
   `scipy.sparse.csr_matrix` — the `occupancy_matrix`, in caller row order.

!!! note "cell_id → grid index"
    `exactextract` numbers cells from the top-left with longitude fastest. geohalo
    converts back to its ascending-latitude convention with
    `row_asc = n_lat - 1 - cell_id // n_lon` and `col = cell_id % n_lon`, so the
    matrix columns line up with a `flat = arr.reshape(-1, n_lat * n_lon)` of the data.

## Polygon row order

Row `i` always corresponds to `geoms.iloc[i]`: `stencil.keys` matches the input
index, and `occupancy_matrix` and `row_sums` use that same order. `ReduceOperator`
and `RestrictedOperator` inherit it, as do NumPy and xarray reduction results.
This also holds when loading from LocalCache or RedisCache, even if a previous
caller requested the same polygons in a different order.

Duplicate labels are kept as separate positional rows, not merged. Prefer unique
keys for label-based selection and hierarchy aggregation. MultiIndex levels and
names are preserved. `BiasTree` retains its own leaves-first node order;
`aggregate_bias_with_tree` aligns xarray reductions to its leaves by key.

!!! warning "Change from 1.2.0"
    Older versions sorted polygon rows by `repr(key)`. Remove any positional
    permutation that previously undid this sort. To explicitly retain that order:

    ```python
    order = np.argsort([repr(key) for key in geoms.index])
    stencil = ghl.Stencil.compute(lats, lons, geoms.iloc[order])
    ```

## What a row looks like

One row of the occupancy matrix is a polygon's **halo** — the weighted set of cells it
overlaps. Interior cells get their full area; boundary cells get a fraction; a hole in
the polygon punches the weights back out.

<figure markdown>
![A polygon's halo of weighted cells](../figures/stencil-halo.png){ width="520" }
<figcaption>
A single stencil row over a 13×14 mesh. Deep cells are fully inside; pale cells
straddle the boundary; the cleared centre is a hole in the polygon. This row, dotted
with the matching cells of the data vector, is the polygon's aggregate.
</figcaption>
</figure>

## `row_sums` and normalisation

`__post_init__` precomputes `row_sums = occupancy_matrix.sum(axis=1)` — each polygon's
total overlap area. That vector is the denominator for `how="mean"`, so the mean hot
path never re-sums the matrix.

## Empty overlaps are errors, not silent zeros

If a polygon does not intersect the grid at all — or its overlap area rounds to zero —
geohalo raises `EmptyOverlapError(geom_key)` rather than emitting an all-zero row that
would later divide to `NaN`. This surfaces grid/polygon mismatches early.

```python
import geohalo as ghl

try:
    stencil = ghl.Stencil.compute(lats, lons, geoms)
except ghl.EmptyOverlapError as e:
    print(f"polygon {e.geom_key} falls outside the grid")
```

## Cost

Building a stencil is the expensive step — it scales with the number of polygons and
their vertex count, since `exactextract` clips every polygon against the raster. The
resulting CSR matrix, by contrast, is tiny (kilobytes to a few megabytes) and is what
you [cache](../guides/caching.md). See the
[`Stencil.compute` rows](../performance.md) in the benchmarks.

Detailed boundaries benefit especially from the WKB input path. Reproduce the
comparison against the previous GeoJSON implementation without downloading data:

```bash
uv run python -m benchmarks.stencil_build
```

Both paths include geometry serialization, coverage extraction, CSR assembly,
and hashing. The benchmark verifies that matrix entries, row sums, polygon keys,
and digests match exactly.

| n_polygons (Brazil L2) | build time | CSR size |
| ---------------------- | ---------- | -------- |
| 50                     | ~21 ms     | 6 KB     |
| 507                    | ~170 ms    | 38 KB    |
| 5571                   | ~2.1 s     | 430 KB   |

Once built, applying it to a batch of 50 grid slices takes a few **milliseconds**.
