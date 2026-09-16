"""Portable artifacts, exact numerical roundtrips, and hostile-input rejection."""

import datetime as dt
import io
import json
import pickle as pk
import subprocess
import sys
import zipfile
from dataclasses import fields, replace
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr

import geohalo as ghl
from geohalo._index_codec import decode_index, decode_scalar, encode_index, encode_scalar


@pytest.fixture(params=[np.float32, np.float64])
def objects(request):
    coords = np.arange(-2., 3.)
    geoms = gpd.GeoSeries([shapely.box(-1.8, -1.8, -.3, .3), shapely.box(.3, -.3, 1.7, 1.6)],
                         index=pd.Index(["zulu", "alpha"], name="zone"))
    stencil = ghl.Stencil.compute(coords, coords, geoms, dtype=request.param, spherical_correction=False)
    operator = ghl.ReduceOperator.compute(stencil, coords, coords, dtype=request.param, iterations=2)
    return [
        stencil, operator, ghl.RestrictedOperator.compute(operator, coords[::-1], 2, 2),
        ghl.BiasTree.compute(pd.DataFrame({"parent": ["root", "root"], "weight": [.3, .7]},
                                         index=geoms.index), weight_col="weight", how="sum"),
        ghl.Resampler.compute(coords, coords, coords[::-1], coords / 2, iterations=2, period=360),
        *[ghl.Resampler.compute(coords, coords, coords[::-1], coords / 2, method="conservative",
                               normalization=normalization, period=360)
          for normalization in ("destination", "covered")],
    ]


def _assert_equal(actual, expected):
    if sp.issparse(expected):
        assert actual.shape == expected.shape
        for field in ("data", "indices", "indptr"):
            _assert_equal(getattr(actual, field), getattr(expected, field))
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(actual, expected, exact=True)
        assert actual.identical(expected)
    elif isinstance(expected, tuple):
        for a, b in zip(actual, expected, strict=True):
            _assert_equal(a, b)
    else:
        assert actual == expected


def _arrays(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return dict(archive)


def _pack(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _change(blob, *, metadata=None, arrays=None, remove=None):
    payload = _arrays(blob)
    if metadata is not None:
        info = json.loads(payload["metadata"].tobytes())
        info.update(metadata)
        payload["metadata"] = np.frombuffer(json.dumps(info).encode(), dtype=np.uint8)
    payload.update(arrays or {})
    if remove is not None:
        del payload[remove]
    return _pack(payload)


def _no_build(*_args, **_kwargs):
    pytest.fail("restoring an artifact must not rebuild it")


def test_all_fields_and_numerics_roundtrip(objects, monkeypatch, tmp_path):
    for obj in objects:
        monkeypatch.setattr(type(obj), "compute", _no_build)
    restored = []
    for obj in objects:
        blob = obj.to_npz()
        file = tmp_path / f"{type(obj).__name__}.npz"
        file.write_bytes(blob)
        actual = type(obj).from_npz(file.read_bytes())
        for field in fields(obj):
            _assert_equal(getattr(actual, field.name), getattr(obj, field.name))
        assert repr(actual) == repr(obj)
        arrays = _arrays(blob)
        assert all(not value.dtype.hasobject for value in arrays.values())
        info = json.loads(arrays["metadata"].tobytes())
        assert info["format"] == "geohalo"
        assert info["version"] == 1
        assert info["kind"] == type(obj).__name__
        assert isinstance(info["geohalo_version"], str)
        assert arrays["digest"].tobytes() == obj.digest
        # Schema compatibility is independent of the writer's package version.
        type(obj).from_npz(_change(blob, metadata={"geohalo_version": "0.0.0"}))
        restored.append(actual)
    values = np.arange(50., dtype=objects[0].occupancy_matrix.dtype).reshape(2, 5, 5)
    values[0, 0, 0] = np.nan
    grid = xr.DataArray(values, dims=("step", "latitude", "longitude"),
                        coords={"latitude": objects[0].lats, "longitude": objects[0].lons})
    for before, after in zip(objects, restored, strict=True):
        if isinstance(before, ghl.Stencil):
            xr.testing.assert_identical(ghl.reduce_with_stencil(grid, after), ghl.reduce_with_stencil(grid, before))
        elif isinstance(before, ghl.ReduceOperator):
            for skipna in (False, True):
                _assert_equal(after.apply_grid(values, how="mean", skipna=skipna, preserve_dtype=True),
                              before.apply_grid(values, how="mean", skipna=skipna, preserve_dtype=True))
        elif isinstance(before, ghl.RestrictedOperator):
            windows = [values[..., ::-1, :][..., rows, cols] for rows, cols in before.windows]
            for skipna in (False, True):
                _assert_equal(after.apply(after.gather(windows), skipna=skipna, preserve_dtype=True),
                              before.apply(before.gather(windows), skipna=skipna, preserve_dtype=True))
        elif isinstance(before, ghl.BiasTree):
            _assert_equal(after.rollup_matrix @ np.array([2., 3.]), before.rollup_matrix @ np.array([2., 3.]))
        else:
            for skipna in ((False, True) if before.method == "conservative" else (False,)):
                _assert_equal(after.apply_grid(values, skipna=skipna), before.apply_grid(values, skipna=skipna))


_KEYS = [
    pd.Index(["z", "á"], name=("zone", 1)), pd.Index(["z", "z"], dtype=object),
    pd.Index([2**64 - 1, 0], dtype="uint64"), pd.Index([.1, np.nan], dtype="float32"),
    pd.Index([1, pd.NA], dtype="Int32"), pd.Index([1., pd.NA], dtype="Float32"),
    pd.Index([True, pd.NA], dtype="boolean"), pd.Index(["z", pd.NA], dtype="string"),
    pd.Index([None, np.nan], dtype=object), pd.Index([pd.NA, pd.NaT], dtype=object),
    pd.Index([b"bytes\x00", "bytes"], dtype=object), pd.Index([True, 1], dtype=object),
    pd.Index([("z", np.int32(2)), ("a", (None, 3))], tupleize_cols=False),
    pd.RangeIndex(9, 5, -2, name="range"),
    pd.CategoricalIndex(["z", None], categories=["unused", "a", "z"], ordered=True, name="cat"),
    pd.MultiIndex(levels=[["unused", "z", "a"], [1, 2, 3]], codes=[[1, 2], [2, -1]], names=["a", ("b", 2)]),
    pd.MultiIndex.from_arrays([["z", "a"]], names=["only"]),
    pd.MultiIndex.from_arrays([pd.Categorical(["z", "a"], categories=["z", "a", "unused"]), [1, 2]]),
    pd.date_range("2020-03-07", periods=2, tz="America/New_York", name="dates"),
    pd.DatetimeIndex(["2020-01-01", None]).as_unit("ms"),
    pd.timedelta_range("1h", periods=2, freq="h", name="deltas"),
    pd.TimedeltaIndex(["1h", None]).as_unit("us"),
    pd.period_range("2020-01", periods=2, freq="M", name="months"),
    pd.IntervalIndex.from_breaks([0., 1., 3.], closed="both", name="intervals"),
    pd.Index([complex(1, 2), complex(3, 4)], name="complex"),
    pd.Index([2**100, -2**100], dtype=object),
]


@pytest.mark.parametrize("keys", _KEYS)
@pytest.mark.parametrize("kind", [ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator, ghl.BiasTree])
def test_key_types_names_order_and_metadata_survive_standalone_exports(objects, keys, kind):
    obj = replace(next(obj for obj in objects if type(obj) is kind), keys=keys)
    if kind is ghl.BiasTree:
        obj = replace(obj, rollup_matrix=sp.eye(2, format="csr"))
    restored = kind.from_npz(obj.to_npz())
    _assert_equal(restored.keys, keys)


@pytest.mark.parametrize("value", [
    None, pd.NA, pd.NaT, True, False, "", "日本語", 2**100, -0., np.nan, np.inf, -np.inf,
    complex(1, -2), b"\x00a\xff", ("x", None, (1, False)),
    np.float32(1.1), np.complex64(2j), np.int16(-2), np.uint64(2**64 - 1), np.bool_(True),
    np.str_("x"), np.str_(""), np.str_("x\x00"), np.str_("\ud800"),
    np.bytes_(b"x"), np.bytes_(b""), np.bytes_(b"x\x00"),
    np.datetime64("2000-01-01", "D"), np.timedelta64(3, "h"),
    pd.Timestamp("2020-01-01T01:00:00.123456789", tz="UTC"), pd.Timedelta("1ns"),
    pd.Period("2020-01", freq="M"), pd.Interval(1, 2),
    dt.date(2000, 1, 1), dt.datetime(2000, 1, 1, tzinfo=dt.UTC, fold=1), dt.timedelta(days=-2, seconds=3),
])
def test_scalar_key_codec(value):
    actual = decode_scalar(json.loads(json.dumps(encode_scalar(value), allow_nan=False)))
    assert type(actual) is type(value)
    if isinstance(value, np.generic):
        assert actual.dtype == value.dtype
        assert actual.tobytes() == value.tobytes()
    _assert_equal(pd.Index([actual], dtype=object), pd.Index([value], dtype=object))


@pytest.mark.parametrize("index", [
    pd.Index([], dtype=object), pd.RangeIndex(0), pd.CategoricalIndex([], categories=["unused"]),
    pd.MultiIndex(levels=[[1], ["unused"]], codes=[[], []]),
    pd.DatetimeIndex([], tz="UTC"), pd.TimedeltaIndex([]), pd.PeriodIndex([], freq="M"),
    pd.IntervalIndex([], dtype="interval[float64]"),
])
def test_empty_index_metadata(index):
    _assert_equal(decode_index(encode_index(index)), index)


def test_csr_roundtrip_keeps_duplicates_explicit_zeros_and_accumulation_order():
    matrix = sp.csr_matrix((np.array([1e16, 1., -1e16, 0.]), np.array([3, 1, 3, 0]), np.array([0, 4])), shape=(1, 4))
    matrix.indices, matrix.indptr = matrix.indices.astype("int64"), matrix.indptr.astype("int64")
    obj = ghl.ReduceOperator(matrix, np.array([1.]), pd.Index(["zone"]), np.arange(2.), np.arange(2.), 1, b"test")
    restored = ghl.ReduceOperator.from_npz(obj.to_npz())
    _assert_equal(restored.matrix, obj.matrix)
    _assert_equal(restored.apply_grid(np.ones((2, 2))), obj.apply_grid(np.ones((2, 2))))


def test_empty_restricted_plan_and_no_cached_properties(objects):
    operator = replace(objects[1], matrix=sp.csr_matrix(objects[1].matrix.shape, dtype=np.float32))
    restricted = ghl.RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    restored = ghl.RestrictedOperator.from_npz(restricted.to_npz())
    assert restored.windows == restored.gathers == ()
    _assert_equal(restored.apply(np.empty((3, 0)), how="sum"), np.zeros((3, 2)))
    for obj in (objects[1], *objects[4:]):
        before = obj.to_npz()
        obj.apply_grid(np.ones((5, 5)))
        assert obj.to_npz() == before


def test_numpy_only_reader(objects, tmp_path):
    file = tmp_path / "operator.npz"
    file.write_bytes(objects[1].to_npz())
    script = """
import json, sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as archive:
    metadata = json.loads(archive['metadata'].tobytes().decode('utf-8'))
    assert metadata['kind'] == 'ReduceOperator'
    assert archive['matrix_shape'].tolist() == [2, 25]
    assert archive['source_lat'].shape == (5,)
    assert archive['row_sums'].shape == (2,)
    assert all(not archive[name].dtype.hasobject for name in archive.files)
assert not any(name.startswith(('geohalo', 'pandas', 'scipy')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(file)], check=True, capture_output=True)


def _executed():
    pytest.fail("unsafe deserialization executed a callable")


class _Unsafe:
    def __reduce__(self):
        return _executed, ()


@pytest.mark.parametrize("cls", [ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator, ghl.BiasTree, ghl.Resampler])
def test_pickle_and_object_arrays_are_never_loaded(objects, cls):
    obj = next(obj for obj in objects if type(obj) is cls)
    raw_pickle = pk.dumps(_Unsafe())
    object_metadata = _pack({"metadata": np.array([_Unsafe()], dtype=object)})
    object_digest = _change(obj.to_npz(), arrays={"digest": np.array([_Unsafe()], dtype=object)})
    for blob in (raw_pickle, object_metadata, object_digest, b"", b"not an archive", obj.to_npz()[:-50]):
        with pytest.raises(ValueError, match="NPZ payload"):
            cls.from_npz(blob)
    with pytest.raises(TypeError, match="expects bytes"):
        cls.from_npz("file.npz")


@pytest.mark.parametrize("metadata", [
    {"version": 999}, {"version": True}, {"format": "foreign"}, {"geohalo_version": None},
    {"keys": {"kind": "custom", "name": ["none"], "values": []}},
    {"keys": {"kind": "index", "dtype": "O", "name": ["__import__", "os"], "values": []}},
    {"iterations": 0}, {"iterations": True},
])
def test_invalid_metadata(objects, metadata):
    with pytest.raises(ValueError, match="NPZ payload"):
        ghl.ReduceOperator.from_npz(_change(objects[1].to_npz(), metadata=metadata))


@pytest.mark.parametrize(("name", "value"), [
    ("matrix_shape", np.array([-1, 25])), ("matrix_shape", np.array([2, 26])),
    ("matrix_shape", np.array([2, 25, 1])), ("matrix_data", np.array(["bad"])),
    ("matrix_indices", np.array([25])), ("matrix_indices", np.array([-1])),
    ("matrix_indptr", np.array([0, 99, 1])), ("matrix_indptr", np.array([])),
    ("matrix_indptr", np.array([1, 1, 1])), ("row_sums", np.ones(3)),
    ("source_lat", np.full(5, np.nan)), ("source_lon", np.ones((1, 5))),
    ("digest", np.arange(3, dtype=np.uint64)), ("metadata", np.ones(3, dtype=np.uint64)),
    ("unexpected", np.array([_Unsafe()], dtype=object)),
])
def test_invalid_array_structure(objects, name, value):
    with pytest.raises(ValueError, match="NPZ payload"):
        ghl.ReduceOperator.from_npz(_change(objects[1].to_npz(), arrays={name: value}))


def test_in_bounds_csr_corruption_is_checked_before_scipy(objects, monkeypatch):
    payload = _arrays(objects[1].to_npz())
    for field, value in (("matrix_indices", -1), ("matrix_indices", 25), ("matrix_indptr", -1)):
        corrupted = {key: array.copy() for key, array in payload.items()}
        corrupted[field][1] = value
        with monkeypatch.context() as patch:
            patch.setattr(sp, "csr_matrix", _no_build)
            with pytest.raises(ValueError, match="CSR structure"):
                ghl.ReduceOperator.from_npz(_pack(corrupted))


def test_wrong_type_missing_fields_duplicate_members_and_npy(objects):
    blob = objects[1].to_npz()
    with pytest.raises(ValueError, match="expected Stencil"):
        ghl.Stencil.from_npz(blob)
    with pytest.raises(ValueError, match="NPZ payload"):
        ghl.ReduceOperator.from_npz(_change(blob, remove="source_lat"))
    stream = io.BytesIO(blob)
    with zipfile.ZipFile(stream, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("digest.npy", b"unused")
    with pytest.raises(ValueError, match="duplicate NPZ"):
        ghl.ReduceOperator.from_npz(stream.getvalue())
    stream = io.BytesIO()
    np.save(stream, np.ones(2))
    with pytest.raises(ValueError, match="expected an NPZ"):
        ghl.ReduceOperator.from_npz(stream.getvalue())


def test_invalid_object_specific_metadata_and_shapes(objects):
    for obj, metadata, arrays in [
        (objects[0], {"spherical_correction": "yes"}, {}),
        (objects[0], {}, {"row_sums": np.zeros(2)}),
        (objects[2], {}, {"windows": np.array([[0, 50, 0, 1]])}),
        (objects[2], {}, {"windows": np.ones((1, 3), dtype=int)}),
        (objects[2], {}, {"lat_chunks": np.array([0, 5])}),
        (objects[2], {}, {"lon_chunks": np.array([1])}),
        (objects[2], {}, {"gather_0": np.array([-1])}),
        (objects[2], {}, {"gather_0": np.array([999])}),
        (objects[3], {"how": "unknown"}, {}),
        (objects[3], {"keys": encode_index(pd.Index(["one"]))}, {}),
        (objects[4], {"method": "unknown"}, {}),
        (objects[4], {"normalization": "covered"}, {}),
        (objects[5], {"normalization": "unknown"}, {}),
        (objects[5], {}, {"latitude_shape": np.array([4, 4])}),
    ]:
        with pytest.raises(ValueError, match="NPZ payload"):
            type(obj).from_npz(_change(obj.to_npz(), metadata=metadata, arrays=arrays))


def test_unsupported_custom_keys_fail_explicitly(objects):
    for key in (object(), frozenset({1, 2})):
        with pytest.raises(TypeError, match="unsupported key type"):
            replace(objects[1], keys=pd.Index([key, "valid"])).to_npz()
    with pytest.raises(TypeError, match="non-object NumPy arrays"):
        replace(objects[1], source_lat=np.array([object()], dtype=object)).to_npz()


def test_unrepresentable_timezone_is_rejected_at_export():
    class CustomZone(dt.tzinfo):
        def utcoffset(self, _value):
            return dt.timedelta(hours=2)

        def dst(self, _value):
            return dt.timedelta(0)

    with pytest.raises(TypeError, match="unsupported timezone"):
        encode_scalar(pd.Timestamp("2020-01-01", tz=CustomZone()))
    with pytest.raises(TypeError, match=r"use pandas\.Timestamp"):
        encode_scalar(dt.datetime(2020, 1, 1, tzinfo=ZoneInfo("Europe/Paris")))


@pytest.mark.parametrize("value", [
    ["numpy", "O", ""], ["numpy", "i4", "00"], ["numpy", "V4", "00000000"],
    ["numpy", "(2,)i4", "0000000000000000"], ["unsupported"],
])
def test_invalid_scalar_encoding(value):
    with pytest.raises(ValueError, match=r"unsupported|invalid"):
        decode_scalar(value)


@pytest.mark.parametrize("metadata", [
    {"kind": "datetime", "dtype": "object", "name": ["none"], "values": []},
    {"kind": "timedelta", "dtype": "int64", "name": ["none"], "values": []},
    {"kind": "index", "dtype": "datetime64[ns]", "name": ["none"], "values": []},
])
def test_invalid_index_encoding(metadata):
    with pytest.raises(ValueError, match=r"invalid|unsupported"):
        decode_index(metadata)


def test_custom_index_and_extension_dtypes_rejected():
    class CustomIndex(pd.Index):
        pass

    with pytest.raises(TypeError, match="unsupported index type"):
        encode_index(CustomIndex._simple_new(np.array([1, 2], dtype=object)))
    with pytest.raises(TypeError, match="unsupported index dtype"):
        encode_index(pd.Index(pd.arrays.SparseArray([0, 1])))


def test_unsupported_serializable_class():
    from geohalo._serialization import NPZSerializable  # noqa: PLC0415

    with pytest.raises(TypeError, match="unsupported NPZ object type"):
        NPZSerializable().to_npz()
    with pytest.raises(TypeError, match="unsupported NPZ object type"):
        NPZSerializable.from_npz(b"")


def test_documented_v1_schema_without_the_geohalo_writer():
    metadata = {
        "format": "geohalo", "version": 1, "kind": "ReduceOperator", "geohalo_version": "1.2.0",
        "iterations": 1,
        "keys": {"kind": "index", "dtype": "|O", "name": ["str", "id"], "values": [["str", "zone"]]},
    }
    blob = _pack({
        "metadata": np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
        "digest": np.arange(32, dtype=np.uint8),
        "matrix_data": np.array([1., 3.], dtype=np.float32),
        "matrix_indices": np.array([0, 3], dtype=np.int32),
        "matrix_indptr": np.array([0, 2], dtype=np.int32),
        "matrix_shape": np.array([1, 4], dtype=np.int64),
        "source_lat": np.array([0., 1.]), "source_lon": np.array([0., 1.]),
        "row_sums": np.array([4.]),
    })
    operator = ghl.ReduceOperator.from_npz(blob)
    assert operator.keys.tolist() == ["zone"]
    assert operator.keys.name == "id"
    assert operator.digest == bytes(range(32))
    values = np.array([[2., 4.], [6., 8.]], dtype=np.float32)
    _assert_equal(operator.apply_grid(values, how="mean", preserve_dtype=True), np.array([6.5], dtype=np.float32))


@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def cache(request, tmp_path):
    if request.param == "local":
        return ghl.LocalCache(tmp_path)
    return ghl.RedisCache(request.getfixturevalue("redis_client"))


def test_cache_ignores_legacy_and_checks_full_digest(cache, objects, tmp_path, monkeypatch):
    operator = objects[1]
    key = operator.digest.hex()[:16]
    malicious = pk.dumps(_Unsafe())
    legacy = tmp_path / "reduceop" / f"{key}.pkl"
    if isinstance(cache, ghl.LocalCache):
        legacy.parent.mkdir()
        legacy.write_bytes(malicious)
    else:
        cache._client.set(f"geohalo:reduceop:v1:{key}", malicious)
    kwargs = {"dtype": operator.matrix.dtype, "iterations": operator.iterations}
    actual = cache.get_or_compute_reduce_operator(objects[0], operator.source_lat, operator.source_lon, **kwargs)
    assert actual.digest == operator.digest
    blob = cache._load("reduceop", key)
    assert json.loads(_arrays(blob)["metadata"].tobytes())["kind"] == "ReduceOperator"
    if isinstance(cache, ghl.LocalCache):
        assert legacy.read_bytes() == malicious
        assert cache._path("reduceop", key).suffix == ".npz"
    else:
        assert cache._client.get(f"geohalo:reduceop:v1:{key}") == malicious
    monkeypatch.setattr(ghl.ReduceOperator, "compute", _no_build)
    cached = cache.get_or_compute_reduce_operator(objects[0], operator.source_lat, operator.source_lon, **kwargs)
    _assert_equal(cached.matrix, actual.matrix)
    cache._store("reduceop", key, _change(blob, arrays={"digest": np.zeros(32, dtype=np.uint8)}))
    with pytest.raises(ValueError, match="digest does not match"):
        cache.get_or_compute_reduce_operator(objects[0], operator.source_lat, operator.source_lon, **kwargs)
    cache._store("reduceop", key, malicious)
    with pytest.raises(ValueError, match="NPZ payload"):
        cache.get_or_compute_reduce_operator(objects[0], operator.source_lat, operator.source_lon, **kwargs)
