"""Allowlisted, JSON-compatible pandas index metadata (never repr/eval)."""

import datetime as dt
import sys
from typing import Any

import numpy as np
import pandas as pd

_EXTENSION_DTYPES = {
    "boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64",
}


def _numpy_dtype(value: str, *, scalar: bool = False) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.fields or dtype.subdtype or dtype.kind not in ("biufcMmSU" if scalar else "biufcOSU"):
        raise ValueError(f"unsupported key dtype: {value!r}")
    return dtype


def _timezone_name(timezone: dt.tzinfo | None) -> str | None:
    if timezone is None:
        return None
    name = str(timezone)
    try:
        pd.Timestamp(0, tz=name)
    except (ValueError, KeyError) as exc:
        raise TypeError("unsupported timezone; use an IANA name or UTC fixed offset") from exc
    return name


def encode_scalar(value: Any) -> list:  # noqa: PLR0912 - explicit allowlist
    """Tagged scalars avoid conflating tuples, bytes, missing values, and text."""
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["NA"]
    if value is pd.NaT:
        return ["NaT"]
    if isinstance(value, np.generic):
        _numpy_dtype(value.dtype.str, scalar=True)
        return ["numpy", value.dtype.str, value.tobytes().hex() if value.dtype.itemsize else ""]
    if type(value) in (str, bool, int):
        return [type(value).__name__, str(value) if type(value) is int else value]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) is tuple:
        return ["tuple", [encode_scalar(item) for item in value]]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", encode_scalar(value.asm8), _timezone_name(value.tz)]
    if isinstance(value, pd.Timedelta):
        return ["timedelta", encode_scalar(value.asm8)]
    if isinstance(value, pd.Period):
        return ["period", value.ordinal, value.freqstr]
    if isinstance(value, pd.Interval):
        return ["interval", encode_scalar(value.left), encode_scalar(value.right), value.closed]
    if type(value) is dt.datetime:
        if value.tzinfo is not None and not isinstance(value.tzinfo, dt.timezone):
            raise TypeError("Python datetime keys require naive/fixed-offset time; use pandas.Timestamp for IANA zones")
        return ["datetime", value.isoformat(), value.fold]
    if type(value) is dt.date:
        return ["date", value.isoformat()]
    if type(value) is dt.timedelta:
        return ["delta", value.days, value.seconds, value.microseconds]
    raise TypeError(f"unsupported key type {type(value).__name__}; use supported scalar or tuple labels")


def decode_scalar(value: Any) -> Any:  # noqa: PLR0912 - explicit allowlist
    match value:
        case ["none"]:
            return None
        case ["NA"]:
            return pd.NA
        case ["NaT"]:
            return pd.NaT
        case ["str", str(text)]:
            return text
        case ["bool", bool(flag)]:
            return flag
        case ["int", str(text)]:
            return int(text)
        case ["float", str(text)]:
            return float.fromhex(text)
        case ["complex", str(real), str(imag)]:
            return complex(float.fromhex(real), float.fromhex(imag))
        case ["bytes", str(text)]:
            return bytes.fromhex(text)
        case ["tuple", list(items)]:
            return tuple(decode_scalar(item) for item in items)
        case ["numpy", str(dtype), str(raw)]:
            dtype = _numpy_dtype(dtype, scalar=True)
            raw = bytes.fromhex(raw)
            if dtype.itemsize == 0 and dtype.kind in "SU" and not raw:
                return dtype.type()
            if len(raw) != dtype.itemsize or not dtype.itemsize:
                raise ValueError("invalid numpy key scalar size")
            # Array scalar extraction strips trailing NULs from S/U values.
            # Construct these scalar types directly to retain the exact label.
            if dtype.kind == "S":
                return np.bytes_(raw)
            if dtype.kind == "U":
                big_endian = dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big")
                return np.str_(raw.decode("utf-32-be" if big_endian else "utf-32-le", errors="surrogatepass"))
            return np.frombuffer(raw, dtype=dtype)[0]
        case ["timestamp", scalar, tz]:
            timestamp = pd.Timestamp(decode_scalar(scalar))
            return timestamp if tz is None else timestamp.tz_localize("UTC").tz_convert(tz)
        case ["timedelta", scalar]:
            return pd.Timedelta(decode_scalar(scalar))
        case ["period", int(ordinal), str(freq)]:
            return pd.Period(ordinal=ordinal, freq=freq)
        case ["interval", left, right, str(closed)]:
            return pd.Interval(decode_scalar(left), decode_scalar(right), closed=closed)
        case ["datetime", str(text), int(fold)]:
            return dt.datetime.fromisoformat(text).replace(fold=fold)
        case ["date", str(text)]:
            return dt.date.fromisoformat(text)
        case ["delta", int(days), int(seconds), int(microseconds)]:
            return dt.timedelta(days=days, seconds=seconds, microseconds=microseconds)
        case _:
            raise ValueError("unsupported key scalar encoding")


def encode_index(index: pd.Index) -> dict:
    if type(index) not in (pd.Index, pd.MultiIndex, pd.RangeIndex, pd.CategoricalIndex, pd.DatetimeIndex,
                           pd.TimedeltaIndex, pd.PeriodIndex, pd.IntervalIndex):
        raise TypeError(f"unsupported index type: {type(index).__name__}")
    if isinstance(index, pd.MultiIndex):
        return {
            "kind": "multi", "levels": [encode_index(level) for level in index.levels],
            "codes": [code.tolist() for code in index.codes], "sortorder": index.sortorder,
            "names": [encode_scalar(name) for name in index.names],
        }
    result = {"name": encode_scalar(index.name)}
    if isinstance(index, pd.RangeIndex):
        return dict(result, kind="range", start=index.start, stop=index.stop, step=index.step)
    if isinstance(index, pd.CategoricalIndex):
        return dict(result, kind="categorical", categories=encode_index(index.categories),
                    codes=index.codes.tolist(), ordered=index.ordered)
    result["values"] = [encode_scalar(value) for value in index]
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        if isinstance(index, pd.DatetimeIndex):
            _timezone_name(index.tz)
        return dict(result, kind="datetime" if isinstance(index, pd.DatetimeIndex) else "timedelta",
                    dtype=str(index.dtype), freq=index.freqstr)
    if isinstance(index, pd.PeriodIndex):
        return dict(result, kind="period", freq=index.freqstr)
    if isinstance(index, pd.IntervalIndex):
        return dict(result, kind="interval", subtype=index.dtype.subtype.str, closed=index.closed)
    dtype = index.dtype
    if isinstance(dtype, np.dtype):
        _numpy_dtype(dtype.str)
        return dict(result, kind="index", dtype=dtype.str)
    if isinstance(dtype, pd.StringDtype):
        return dict(result, kind="string", storage=dtype.storage,
                    na_value=encode_scalar(getattr(dtype, "na_value", pd.NA)))
    if str(dtype) in _EXTENSION_DTYPES:
        return dict(result, kind="extension", dtype=str(dtype))
    raise TypeError(f"unsupported index dtype: {dtype}")


def decode_index(payload: dict) -> pd.Index:  # noqa: PLR0912 - explicit allowlist
    kind = payload["kind"]
    if kind == "multi":
        return pd.MultiIndex(
            levels=[decode_index(level) for level in payload["levels"]], codes=payload["codes"],
            names=[decode_scalar(name) for name in payload["names"]], sortorder=payload["sortorder"],
            verify_integrity=True,
        )
    name = decode_scalar(payload["name"])
    if kind == "range":
        return pd.RangeIndex(payload["start"], payload["stop"], payload["step"], name=name)
    if kind == "categorical":
        categories = pd.Categorical.from_codes(
            payload["codes"], categories=decode_index(payload["categories"]), ordered=payload["ordered"],
        )
        return pd.CategoricalIndex(categories, name=name)
    values = [decode_scalar(value) for value in payload["values"]]
    if kind == "datetime":
        dtype = payload["dtype"]
        if not isinstance(dtype, str) or not dtype.startswith("datetime64["):
            raise ValueError("invalid datetime index dtype")
        return pd.DatetimeIndex(values, dtype=dtype, freq=payload["freq"], name=name)
    if kind == "timedelta":
        dtype = np.dtype(payload["dtype"])
        if dtype.kind != "m":
            raise ValueError("invalid timedelta index dtype")
        return pd.TimedeltaIndex(values, dtype=dtype, freq=payload["freq"], name=name)
    if kind == "period":
        return pd.PeriodIndex(values, freq=payload["freq"], name=name)
    if kind == "interval":
        dtype = pd.IntervalDtype(_numpy_dtype(payload["subtype"]), payload["closed"])
        return pd.IntervalIndex(values, dtype=dtype, name=name)
    if kind == "index":
        dtype = _numpy_dtype(payload["dtype"])
    elif kind == "extension" and payload["dtype"] in _EXTENSION_DTYPES:
        dtype = payload["dtype"]
    elif kind == "string":
        na_value = decode_scalar(payload["na_value"])
        # pandas 2.x has no na_value keyword and always uses pd.NA.
        options = {} if na_value is pd.NA else {"na_value": na_value}
        dtype = pd.StringDtype(storage=payload["storage"], **options)
    else:
        raise ValueError(f"unsupported index encoding: {kind!r}")
    return pd.Index(values, dtype=dtype, name=name, tupleize_cols=False)
