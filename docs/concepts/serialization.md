# Portable operator artifacts

`Stencil`, `ReduceOperator`, `Resampler`, `RestrictedOperator`, and `BiasTree`
all expose the same byte-oriented API:

```python
from pathlib import Path
import geohalo as ghl

# Build once, then store in a file, database bytea column, or object store.
operator = ghl.ReduceOperator.compute(stencil, source_lat, source_lon)
blob: bytes = operator.to_npz()
Path("operator.npz").write_bytes(blob)

# Restore without geometry extraction, resampler construction, or fusion.
restored = ghl.ReduceOperator.from_npz(Path("operator.npz").read_bytes())
result = restored.apply_grid(values, how="mean")
```

`to_npz()` returns bytes; `from_npz(blob)` accepts bytes, not a filename or an
open file. Use the class matching the stored object. Wrong classes, unknown
schema versions, missing fields, object arrays, and malformed structures raise
`ValueError`. Unsupported custom key types raise `TypeError` when exporting.

Both cache backends use this same format. Standalone exports retain the object's
**current row order**, unlike canonical cache storage; always interpret rows
using the restored `keys`. Digests identify build inputs, not row permutations.

## What survives a roundtrip

- CSR coefficient and index arrays, including dtypes, accumulation order,
  duplicate entries, and explicit zeros; serialization never sorts or merges
  matrix entries.
- Source/target coordinates in their stored order, input digests, normalizers,
  and the object's apply-time configuration.
- Restricted windows, per-window gather positions, and spatial chunk sizes.
- Both resampler representations: the mean-preserving full CSR transform or
  conservative latitude/longitude factors plus normalization. Conservative
  exports never materialize a grid-to-grid matrix.
- Index names, dtypes, duplicates, order, MultiIndex levels/codes (including
  unused levels and missing codes), categorical categories/order, and date/time
  index frequency and timezone metadata.

Apply-time caches such as compact matrix views are not serialized. They are
rebuilt lazily from the restored arrays. Input geometries and original tree
edges are not included: these are **application-ready operators**, not complete
recipes for rebuilding them. Options already baked into coefficients (such as
resampler period and cell bounds) are represented by those coefficients and the
input digest, not duplicated as a build recipe.

## Supported keys

Flat indexes support NumPy numeric/object dtypes and pandas nullable numeric,
boolean, and string dtypes. RangeIndex, CategoricalIndex, MultiIndex,
DatetimeIndex, TimedeltaIndex, PeriodIndex, and IntervalIndex retain their
specialized metadata. Reading a pandas string index with Arrow storage requires
the same optional Arrow dependency as constructing that index normally.
Likewise, pandas 3's NaN-based `str` dtype needs pandas 3 to restore that exact
dtype; older pandas can still be used with object indexes or nullable `string`
indexes. Unsupported reader-side dtype features fail rather than silently
changing labels or missing-value semantics.

Object labels and index names use a fixed typed encoding for Python strings,
integers, floats, complex numbers, booleans, bytes, tuples (recursively), `None`,
`pd.NA`, `pd.NaT`, NumPy numeric/string/date/time scalars, Python dates,
naive/fixed-offset datetimes and timedeltas, and pandas Timestamp, Timedelta, Period and Interval
scalars. Non-finite numbers are encoded as tagged strings, not nonstandard JSON
numbers. Integers use decimal strings, preserving values beyond JavaScript's
integer precision. NumPy scalars include their dtype and hex-encoded bytes.
Pandas date/time keys support IANA timezone names and UTC fixed offsets; convert
custom timezone implementations to one of those before exporting. Use pandas
Timestamp rather than Python datetime labels for IANA zones.

Arbitrary Python objects, custom Index subclasses, and unlisted extension dtypes
are deliberately unsupported. Convert such labels to supported values before
building the operator. There is no implicit `repr` conversion, dynamic class
import, `eval`, or fallback to executable object serialization.

## NPZ schema version 1

The container is an uncompressed NumPy NPZ archive: named `.npy` arrays in a ZIP
file. Every array is non-object. It can be inspected with NumPy and the standard
library alone, without importing geohalo, pandas, SciPy, or exactextract:

```python
import io
import json
import numpy as np

with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
    metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    print(metadata["kind"], metadata["version"], metadata["geohalo_version"])
    print(archive["digest"].tobytes().hex())
    data = archive["matrix_data"]       # ReduceOperator example
    indices = archive["matrix_indices"]
    indptr = archive["matrix_indptr"]
    shape = tuple(archive["matrix_shape"])
```

This uses NumPy's documented
[`allow_pickle=False` loading mode](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
Reading the arrays is independent of Python class layout; restoring a geohalo
object and its pandas index still requires the relevant libraries.

### Common members

`metadata` is a one-dimensional uint8 array of UTF-8 JSON bytes. Required fields:

| Field | Meaning |
| --- | --- |
| `format` | The string `"geohalo"` |
| `version` | Integer NPZ schema version, currently `1` |
| `kind` | `Stencil`, `ReduceOperator`, `Resampler`, `RestrictedOperator`, or `BiasTree` |
| `geohalo_version` | Writer's package version; informational, not a compatibility gate |
| `keys` | Typed index metadata for every object except `Resampler` |

`digest` is a one-dimensional uint8 array holding the object's digest bytes.
Builder-produced digests are SHA-256 (32 bytes). Each CSR matrix named `prefix`
uses `prefix_data`, `prefix_indices`, `prefix_indptr`, and `prefix_shape`.
`shape` is a two-element int64 array; coefficient and index buffers retain their
original dtypes. CSR indices are zero-based.

### Object-specific members

| Kind | Array members beyond `metadata` and `digest` | Extra JSON fields |
| --- | --- | --- |
| Stencil | `matrix_*`, `row_sums`, `lats`, `lons` | `spherical_correction`, `partial_cell_weighting` |
| ReduceOperator | `matrix_*`, `row_sums`, `source_lat`, `source_lon` | `iterations` |
| RestrictedOperator | `matrix_*`, `row_sums`, `source_lat`, `source_lon`, `windows`, `gather_0` … `gather_N`, `lat_chunks`, `lon_chunks` | — |
| BiasTree | `matrix_*` | `how` |
| Mean-preserving Resampler | `matrix_*`, `source_lat`, `source_lon`, `target_lat`, `target_lon` | `method="meanpreserving"`, `normalization="destination"` |
| Conservative Resampler | `latitude_*`, `longitude_*`, `source_lat`, `source_lon`, `target_lat`, `target_lon` | `method="conservative"`, `normalization` |

Stencil `partial_cell_weighting` is `"approximate"` or `"exact"`. Older version-1
archives without this field load as `"approximate"`. Exact weighting requires
`spherical_correction=true`. Matrices and their stored normalizers remain the
application data; loading never reconstructs polygon intersections.

Restricted `windows` is an `(n_windows, 4)` int64 array with rows
`[latitude_start, latitude_stop, longitude_start, longitude_stop]`. Bounds are
half-open, with unit steps, in stored source order. `gather_i` is a 1-D integer
array of C-order positions within window `i`; concatenating the gathered arrays
defines compact matrix column order. No windows means shape `(0, 4)` and no
`gather_i` members. Chunk arrays list consecutive positive sizes along each axis.

Key metadata is tagged JSON. Flat indexes have `kind`, a tagged `name`, tagged
`values`, and a dtype descriptor. Specialized kinds are `range`, `categorical`,
`multi`, `datetime`, `timedelta`, `period`, and `interval`; MultiIndex stores
recursively encoded `levels`, integer `codes`, tagged `names`, and `sortorder`.
For example, a tuple label `("zone", 7)` is
`["tuple", [["str", "zone"], ["int", "7"]]]`, not a string representation to
evaluate. The allowlisted codec is defined in `geohalo._index_codec`.

The schema version is separate from geohalo's release version and from legacy
cache payload versions. Readers reject unknown schema versions rather than
guessing. Future incompatible format changes require a new schema version;
support for older schemas must be explicit. This is not a promise that future
versions of every dependency will retain every pandas dtype or timezone name.

## Validation and trust

Loading always disables pickle. The reader checks object kind/version, required
arrays and dtypes, CSR bounds and monotonic pointers, matrix/grid/key dimensions,
normalizer lengths, and restricted chunk/window/gather bounds before use.
Unexpected or duplicate archive members are rejected. Invalid NPZ cache hits
raise errors; `force_recompute=True` explicitly replaces them.

This prevents the executable-object loading inherent in pickle. It does **not**
authenticate a writer, verify geometric correctness, or impose an archive size
limit. A validly shaped matrix can still contain wrong coefficients, and a large
or compressed archive can exhaust memory. Authenticate shared artifacts, limit
input sizes/resources, and maintain dependency security updates. Use an external
checksum or signature if you need integrity/provenance checks. The embedded
input digest is not a checksum of the archive contents.

## Migrating existing caches

LocalCache now reads/writes `path/<namespace>/<key>.npz`. RedisCache uses
`geohalo:<namespace>:npz:v1:<key>`. Existing `.pkl` files and old Redis prefixes
are neither read nor deleted. The first request rebuilds each operator once and
writes NPZ; old and new deployments can coexist without reading each other's
formats. Input digests themselves are unchanged.

There is no automatic legacy loader or migration fallback. For externally
managed artifacts, export `to_npz()` from an already available object using this
version, or rebuild from inputs. Remove obsolete cache entries yourself after
confirming older deployments no longer need them.
