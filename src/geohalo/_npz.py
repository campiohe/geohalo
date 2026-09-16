"""Version-1 portable operator archives. See docs/concepts/serialization.md."""

import io
import json
import zipfile
from importlib.metadata import version

import numpy as np
import pandas as pd
import scipy.sparse as sp

from geohalo._index_codec import decode_index, encode_index
from geohalo._serialization import NPZSerializable
from geohalo.bias_tree import BiasTree
from geohalo.reduce_operator import ReduceOperator
from geohalo.resampler import Resampler
from geohalo.restricted_operator import RestrictedOperator
from geohalo.stencil import Stencil

FORMAT_VERSION = 1
_TYPES = (Stencil, ReduceOperator, Resampler, RestrictedOperator, BiasTree)


def _put_matrix(arrays: dict, prefix: str, matrix: sp.csr_matrix) -> None:
    arrays.update({f"{prefix}_{field}": getattr(matrix, field) for field in ("data", "indices", "indptr")})
    arrays[f"{prefix}_shape"] = np.asarray(matrix.shape, dtype=np.int64)


def encode(obj: NPZSerializable) -> bytes:  # noqa: PLR0912 - five explicit object layouts
    if type(obj) not in _TYPES:
        raise TypeError(f"unsupported NPZ object type: {type(obj).__name__}")
    metadata = {"format": "geohalo", "version": FORMAT_VERSION, "kind": type(obj).__name__,
                "geohalo_version": version("geohalo")}
    arrays = {"digest": np.frombuffer(obj.digest, dtype=np.uint8)}
    if hasattr(obj, "keys"):
        metadata["keys"] = encode_index(obj.keys)
    if isinstance(obj, Stencil):
        _put_matrix(arrays, "matrix", obj.occupancy_matrix)
        arrays.update(lats=obj.lats, lons=obj.lons, row_sums=obj.row_sums)
        metadata["spherical_correction"] = bool(obj.spherical_correction)
    elif isinstance(obj, BiasTree):
        _put_matrix(arrays, "matrix", obj.rollup_matrix)
        metadata["how"] = obj.how
    else:
        arrays.update(source_lat=obj.source_lat, source_lon=obj.source_lon)
        if isinstance(obj, Resampler):
            arrays.update(target_lat=obj.target_lat, target_lon=obj.target_lon)
            metadata.update(method=obj.method, normalization=obj.normalization)
            if obj.axis_weights is None:
                _put_matrix(arrays, "matrix", obj.transform_matrix)
            else:
                for prefix, matrix in zip(("latitude", "longitude"), obj.axis_weights, strict=True):
                    _put_matrix(arrays, prefix, matrix)
        else:
            _put_matrix(arrays, "matrix", obj.matrix)
            arrays["row_sums"] = obj.row_sums
            if isinstance(obj, ReduceOperator):
                metadata["iterations"] = int(obj.iterations)
            else:
                arrays.update(
                    windows=np.asarray([(r.start, r.stop, c.start, c.stop) for r, c in obj.windows],
                                       dtype=np.int64).reshape(-1, 4),
                    lat_chunks=np.asarray(obj.lat_chunks, dtype=np.int64),
                    lon_chunks=np.asarray(obj.lon_chunks, dtype=np.int64),
                )
                for i, gather in enumerate(obj.gathers):
                    arrays[f"gather_{i}"] = gather
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    # Checking every array also supports NumPy versions predating savez's
    # allow_pickle keyword (where that keyword would become an archive member).
    if any(not isinstance(array, np.ndarray) or array.dtype.hasobject for array in arrays.values()):
        raise TypeError("NPZ payloads require non-object NumPy arrays")
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


class _Reader:
    def __init__(self, archive: np.lib.npyio.NpzFile) -> None:
        self.archive = archive
        self.used = set()

    def array(self, name: str, *, kinds: str, ndim: int = 1) -> np.ndarray:
        value = self.archive[name]
        self.used.add(name)
        if not isinstance(value, np.ndarray) or value.dtype.kind not in kinds or value.ndim != ndim:
            raise ValueError(f"invalid array {name!r}: expected {ndim} dimensions and dtype kind in {kinds}")
        return value

    def matrix(self, prefix: str, *, shape: tuple[int, int] | None = None) -> sp.csr_matrix:
        dimensions = self.array(f"{prefix}_shape", kinds="iu")
        if dimensions.shape != (2,) or np.any(dimensions < 0) or np.any(dimensions > np.iinfo(np.intp).max):
            raise ValueError(f"invalid {prefix} CSR shape")
        dimensions = tuple(map(int, dimensions))
        if shape is not None and dimensions != shape:
            raise ValueError(f"{prefix} CSR shape does not match keys/grid")
        data = self.array(f"{prefix}_data", kinds="fc")
        indices = self.array(f"{prefix}_indices", kinds="i")
        indptr = self.array(f"{prefix}_indptr", kinds="i")
        if (
            indptr.size != dimensions[0] + 1 or indices.size != data.size
            or indptr[0] != 0 or indptr[-1] != data.size or np.any(indptr[1:] < indptr[:-1])
            or np.any(indices < 0) or np.any(indices >= dimensions[1])
            or indices.dtype.itemsize not in (4, 8) or indptr.dtype.itemsize not in (4, 8)
        ):
            raise ValueError(f"invalid {prefix} CSR structure")
        matrix = sp.csr_matrix((data, indices, indptr), shape=dimensions)
        # SciPy may downcast int64 indices on construction. Preserve the stored
        # triples exactly, including unsorted entries, duplicates, and zeros.
        matrix.indices, matrix.indptr = indices, indptr
        return matrix

    def coordinates(self, *names: str) -> dict:
        values = {name: self.array(name, kinds="fiu") for name in names}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("grid coordinates must be finite")
        return values

    def row_sums(self, count: int) -> np.ndarray:
        values = self.array("row_sums", kinds="fc")
        if values.size != count:
            raise ValueError("row_sums must match matrix rows")
        return values

    def finish(self) -> None:
        if self.used != set(self.archive.files):
            raise ValueError("unexpected NPZ archive members")


def _restricted(reader: _Reader, coords: dict, keys: pd.Index, digest: bytes) -> RestrictedOperator:
    windows = reader.array("windows", kinds="i", ndim=2)
    if windows.shape[1] != 4:
        raise ValueError("windows must have four columns")
    chunks = {name: reader.array(name, kinds="i") for name in ("lat_chunks", "lon_chunks")}
    for name, axis in (("lat_chunks", "source_lat"), ("lon_chunks", "source_lon")):
        sizes = chunks[name]
        if np.any(sizes <= 0) or sum(map(int, sizes)) != coords[axis].size:
            raise ValueError("chunk sizes must be positive and cover the source axis")
    gathers, slices = [], []
    for i, (r0, r1, c0, c1) in enumerate(windows.tolist()):
        if not (0 <= r0 < r1 <= coords["source_lat"].size and 0 <= c0 < c1 <= coords["source_lon"].size):
            raise ValueError("window bounds must lie within the source grid")
        gather = reader.array(f"gather_{i}", kinds="i")
        if np.any(gather < 0) or np.any(gather >= (r1 - r0) * (c1 - c0)):
            raise ValueError("gather positions must lie within their window")
        gathers.append(gather)
        slices.append((slice(r0, r1), slice(c0, c1)))
    matrix = reader.matrix("matrix", shape=(len(keys), sum(gather.size for gather in gathers)))
    return RestrictedOperator(
        matrix=matrix, row_sums=reader.row_sums(len(keys)), keys=keys, digest=digest,
        windows=tuple(slices), gathers=tuple(gathers),
        **{name: tuple(map(int, sizes)) for name, sizes in chunks.items()}, **coords,
    )


def _restore[T: NPZSerializable](reader: _Reader, metadata: dict, cls: type[T]) -> T:  # noqa: PLR0912
    digest = reader.array("digest", kinds="u")
    if digest.dtype != np.uint8:
        raise ValueError("digest must be uint8 bytes")
    digest = digest.tobytes()
    keys = None if cls is Resampler else decode_index(metadata["keys"])
    if cls is Stencil:
        coords = reader.coordinates("lats", "lons")
        matrix = reader.matrix("matrix", shape=(len(keys), coords["lats"].size * coords["lons"].size))
        if type(metadata["spherical_correction"]) is not bool:
            raise ValueError("spherical_correction must be boolean")
        obj = Stencil(matrix, keys, **coords, digest=digest, spherical_correction=metadata["spherical_correction"])
        row_sums = reader.row_sums(len(keys))
        if not np.array_equal(row_sums, obj.row_sums, equal_nan=True):
            raise ValueError("stencil row_sums do not match its coefficients")
        return obj
    if cls is BiasTree:
        matrix = reader.matrix("matrix")
        if matrix.shape[0] != len(keys) or matrix.shape[1] > len(keys):
            raise ValueError("tree CSR shape does not match keys/leaves")
        if metadata["how"] not in ("mean", "sum"):
            raise ValueError("invalid tree reduction mode")
        return BiasTree(matrix, keys, digest, metadata["how"])
    coords = reader.coordinates("source_lat", "source_lon")
    if cls is RestrictedOperator:
        return _restricted(reader, coords, keys, digest)
    source_size = coords["source_lat"].size * coords["source_lon"].size
    if cls is ReduceOperator:
        iterations = metadata["iterations"]
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        return ReduceOperator(
            reader.matrix("matrix", shape=(len(keys), source_size)), reader.row_sums(len(keys)),
            keys, **coords, iterations=iterations, digest=digest,
        )
    coords.update(reader.coordinates("target_lat", "target_lon"))
    method, normalization = metadata["method"], metadata["normalization"]
    if normalization not in ("destination", "covered"):
        raise ValueError("invalid resampler normalization")
    if method == "conservative":
        axes = tuple(reader.matrix(prefix, shape=(coords[target].size, coords[source].size))
                     for prefix, source, target in (("latitude", "source_lat", "target_lat"),
                                                    ("longitude", "source_lon", "target_lon")))
        return Resampler(None, **coords, digest=digest, axis_weights=axes, normalization=normalization)
    if method != "meanpreserving" or normalization != "destination":
        raise ValueError("invalid resampler method/normalization")
    target_size = coords["target_lat"].size * coords["target_lon"].size
    return Resampler(reader.matrix("matrix", shape=(target_size, source_size)), **coords, digest=digest)


def _decode_archive(archive: np.lib.npyio.NpzFile, cls: type[NPZSerializable]) -> NPZSerializable:
    if len(archive.files) != len(set(archive.files)):
        raise ValueError("duplicate NPZ archive members")
    reader = _Reader(archive)
    raw = reader.array("metadata", kinds="u")
    if raw.dtype != np.uint8:
        raise ValueError("metadata must be UTF-8 uint8 bytes")
    metadata = json.loads(raw.tobytes().decode("utf-8"))
    if metadata["format"] != "geohalo":
        raise ValueError("not a geohalo archive")
    if type(metadata["version"]) is not int or metadata["version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported NPZ schema version: {metadata['version']!r}")
    if metadata["kind"] != cls.__name__:
        raise ValueError(f"expected {cls.__name__}, found {metadata['kind']!r}")
    if not isinstance(metadata["geohalo_version"], str):
        raise TypeError("missing writer version")
    result = _restore(reader, metadata, cls)
    reader.finish()
    return result


def decode[T: NPZSerializable](blob: bytes, cls: type[T]) -> T:
    if not isinstance(blob, bytes):
        raise TypeError("from_npz expects bytes; use Path.read_bytes() for files")
    if cls not in _TYPES:
        raise TypeError(f"unsupported NPZ object type: {cls.__name__}")
    try:
        archive = np.load(io.BytesIO(blob), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise TypeError("expected an NPZ archive")  # noqa: TRY301 - normalize external format errors
        with archive:
            return _decode_archive(archive, cls)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid {cls.__name__} NPZ payload: {exc}") from exc
