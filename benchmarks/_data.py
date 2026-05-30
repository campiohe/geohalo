"""Shared fixtures: GADM fetch, polygon construction, grid, hierarchies.

Used by `benchmarks/run.py`.
"""

import datetime as dt
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

GADM_JSON_BASE = "https://geodata.ucdavis.edu/gadm/gadm4.1/json"
ECMWF_BUCKET_URL = "https://ecmwf-forecasts.s3.amazonaws.com"

# Sovereign states of the Americas (35) — ISO3.
AMERICAS_ISO3 = (
    "ATG", "ARG", "BHS", "BRB", "BLZ", "BOL", "BRA", "CAN", "CHL", "COL",
    "CRI", "CUB", "DMA", "DOM", "ECU", "SLV", "GRD", "GTM", "GUY", "HTI",
    "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT",
    "SUR", "TTO", "USA", "URY", "VEN",
)

# European countries (~44) — ISO3.
EUROPE_ISO3 = (
    "ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CZE", "DNK",
    "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ITA", "XKO",
    "LVA", "LIE", "LTU", "LUX", "MLT", "MDA", "MCO", "MNE", "NLD", "MKD",
    "NOR", "POL", "PRT", "ROU", "SMR", "SRB", "SVK", "SVN", "ESP", "SWE",
    "CHE", "UKR", "GBR", "RUS",
)


def select_enfo_messages(idx_lines: list[str], params: tuple[str, ...], members: tuple[int, ...]) -> list[dict]:
    """Filter ENS GRIB `.index` JSON lines to the requested params+members, sorted by byte offset."""
    member_strs = {str(m) for m in members}
    param_set = set(params)
    selected = [
        e
        for e in (json.loads(line) for line in idx_lines)
        if e["param"] in param_set and e.get("number") in member_strs
    ]
    selected.sort(key=lambda e: e["_offset"])
    return selected


def latest_enfo_cycle(required_steps_h: tuple[int, ...]) -> tuple[str, str]:
    """Most recent date/cycle on the bucket whose `.index` exists for every required step."""
    now = dt.datetime.now(dt.UTC)
    for back in range(7):
        day = now - dt.timedelta(days=back)
        for cycle_hour in (12, 0):
            date_str = day.strftime("%Y%m%d")
            cycle = f"{cycle_hour:02d}z"
            ts = f"{date_str}{cycle_hour:02d}0000"
            if all(_enfo_index_exists(date_str, cycle, ts, step_h) for step_h in required_steps_h):
                return date_str, cycle
    raise RuntimeError(f"no ECMWF enfo cycle with steps {required_steps_h} found in last 7 days")


def _enfo_index_exists(date_str: str, cycle: str, ts: str, step_h: int, *, retries: int = 3) -> bool:
    url = f"{ECMWF_BUCKET_URL}/{date_str}/{cycle}/ifs/0p25/enfo/{ts}-{step_h}h-enfo-ef.index"
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                return resp.status == 200
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):  # rate-limit / transient server error: back off
                if attempt == retries - 1:
                    print(f"  HEAD {url} got {exc.code} after {retries} tries; treating as absent")
                    return False
                time.sleep(2.0 * (attempt + 1))
                continue
            return False  # definitive (e.g. 404): this cycle/step is not published
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Transient (connection reset, DNS, SSL handshake, timeout): retry before giving up.
            if attempt == retries - 1:
                print(f"  HEAD {url} failed after {retries} tries ({exc}); treating as absent")
                return False
            time.sleep(1.0 + attempt)
    return False


def fetch_enfo_step_grib(
    date_str: str, cycle: str, step_h: int, params: tuple[str, ...],
    members: tuple[int, ...], cache_dir: Path,
) -> Path:
    """Range-read the requested messages of one ENS step file into a concatenated GRIB2.

    GRIB2 messages are self-delimiting, so concatenating raw bytes yields a valid file.
    """
    ts = f"{date_str}{cycle[:2]}0000"
    base = f"{ECMWF_BUCKET_URL}/{date_str}/{cycle}/ifs/0p25/enfo/{ts}-{step_h}h-enfo-ef"
    member_tag = f"m{members[0]}-{members[-1]}"
    cache_path = cache_dir / f"enfo_{date_str}_{cycle}_{step_h}h_{'-'.join(params)}_{member_tag}.grib2"
    if cache_path.exists():
        return cache_path

    print(f"fetching ENS index {date_str} {cycle} step {step_h}h ...")
    with urllib.request.urlopen(base + ".index", timeout=60) as resp:  # noqa: S310
        idx_lines = resp.read().decode().splitlines()
    selected = select_enfo_messages(idx_lines, params, members)
    if not selected:
        raise RuntimeError(f"no messages match params={params} members={members} step={step_h}h")
    total_mb = sum(e["_length"] for e in selected) / 1e6
    print(f"  range-reading {len(selected)} messages ({total_mb:.1f} MB) -> {cache_path.name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with tmp.open("wb") as out:
            for entry in selected:
                start = entry["_offset"]
                end = start + entry["_length"] - 1
                req = urllib.request.Request(base + ".grib2", headers={"Range": f"bytes={start}-{end}"})  # noqa: S310
                with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                    out.write(resp.read())
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return cache_path


ENFO_STEPS_H = (6, 12, 18, 24)
ENFO_MEMBERS = tuple(range(1, 51))  # 50 perturbed members


def fetch_ecmwf_global(
    cache_dir: Path,
    *,
    members: tuple[int, ...] = ENFO_MEMBERS,
    steps_h: tuple[int, ...] = ENFO_STEPS_H,
    short_name: str = "2t",
) -> xr.DataArray:
    """Real whole-world IFS ENS 0.25 degree field as (member, step, latitude, longitude).

    Auto-detects the latest cycle holding all `steps_h`; byte-range fetches every
    member x step `short_name` message and concatenates. Cached netCDF per (cycle, members, steps).
    """
    steps_tag = "-".join(map(str, steps_h))
    suffix = f"_{short_name}_m{members[0]}-{members[-1]}_s{steps_tag}.nc"
    # Cache-first: any resolved-cycle file for this (var, members, steps) loads with no network.
    # Cycle auto-detection (and the whole S3 dependency) only runs on a cold cache — so repeated
    # benchmark runs don't hammer the bucket (which rate-limits with HTTP 503).
    cached = sorted(cache_dir.glob(f"ecmwf_global_*{suffix}"))
    if cached:
        parts = cached[-1].stem.split("_")  # ecmwf_global_{date}_{cycle}_{var}_...
        return xr.open_dataarray(cached[-1]).assign_attrs(ecmwf_cycle=f"{parts[2]} {parts[3]}")

    date_str, cycle = latest_enfo_cycle(steps_h)
    nc_path = cache_dir / f"ecmwf_global_{date_str}_{cycle}{suffix}"
    if nc_path.exists():
        return xr.open_dataarray(nc_path).assign_attrs(ecmwf_cycle=f"{date_str} {cycle}")

    per_step: list[xr.DataArray] = []
    for step_h in steps_h:
        grib = fetch_enfo_step_grib(date_str, cycle, step_h, (short_name,), members, cache_dir)
        ds = xr.open_dataset(
            grib, engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": short_name}, "indexpath": ""},
        )
        da = ds[next(iter(ds.data_vars))].rename({"number": "member"})
        da = da.expand_dims(step=[step_h])
        # Drop scalar coords (time, valid_time, surface, ...) that differ per step:
        # valid_time = time + step, so concat below would otherwise hit a MergeError.
        # Only member/step/latitude/longitude matter; reset_coords drops the rest anyway.
        da = da.drop_vars([c for c in da.coords if c not in da.dims])
        per_step.append(da)
    # coords="minimal": lat/lon/member are identical across steps, so don't compare/concat them
    # (only the "step" coord varies). Silences xarray's coords-default FutureWarning.
    out = xr.concat(per_step, dim="step", coords="minimal").transpose("member", "step", "latitude", "longitude")
    out = out.reset_coords(drop=True).assign_attrs(ecmwf_cycle=f"{date_str} {cycle}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(nc_path)
    return out


def fetch_gadm(iso3: str, level: int, cache_dir: Path) -> dict:
    """Download (and cache) a GADM 4.1 GeoJSON for one country at one admin level."""
    json_path = cache_dir / f"gadm41_{iso3}_{level}.json"
    if not json_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / f"gadm41_{iso3}_{level}.json.zip"
        url = f"{GADM_JSON_BASE}/gadm41_{iso3}_{level}.json.zip"
        print(f"downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".json"))
            json_path.write_bytes(zf.read(inner))
        zip_path.unlink()
    return json.loads(json_path.read_text())


def polygons_from_gadm(geojson: dict, level: int) -> gpd.GeoSeries:
    """GeoSeries keyed by the level's GID. Level 0 -> "GID_0" string key; level >= 1 ->
    tuple ("GID_1", ..., "GID_{level}")."""
    keys: list = []
    geoms: list = []
    for feature in geojson["features"]:
        props = feature["properties"]
        if level == 0:
            key = props["GID_0"]
        else:
            key = tuple(props[f"GID_{lvl}"] for lvl in range(1, level + 1))
        keys.append(key)
        geoms.append(shapely.geometry.shape(feature["geometry"]))
    index = pd.Index(keys, name="gid", tupleize_cols=False)
    return gpd.GeoSeries(geoms, index=index)


def _countries(iso3_list: tuple[str, ...], cache_dir: Path) -> gpd.GeoSeries:
    parts = [polygons_from_gadm(fetch_gadm(iso3, 0, cache_dir), level=0) for iso3 in iso3_list]
    return gpd.GeoSeries(pd.concat(parts))


def brazil_municipalities(cache_dir: Path) -> gpd.GeoSeries:
    return polygons_from_gadm(fetch_gadm("BRA", 2, cache_dir), level=2)


def us_counties(cache_dir: Path) -> gpd.GeoSeries:
    return polygons_from_gadm(fetch_gadm("USA", 2, cache_dir), level=2)


def brazil_country(cache_dir: Path) -> gpd.GeoSeries:
    return polygons_from_gadm(fetch_gadm("BRA", 0, cache_dir), level=0)


def americas_countries(cache_dir: Path) -> gpd.GeoSeries:
    return _countries(AMERICAS_ISO3, cache_dir)


def americas_europe_countries(cache_dir: Path) -> gpd.GeoSeries:
    return _countries(AMERICAS_ISO3 + EUROPE_ISO3, cache_dir)


def build_brazil_hierarchy(geoms: gpd.GeoSeries, *, depth: int = 3, country_key: str = "BRA") -> pd.DataFrame:
    """Edges (index=child, `parent` column) for the GADM Brazil hierarchy.

    depth=2: municipality (GID_1, GID_2) -> state GID_1.
    depth=3: also state GID_1 -> country root `country_key`.
    """
    if depth not in (2, 3):
        raise ValueError(f"depth must be 2 or 3, got {depth}")
    munis = list(geoms.index)
    states = [k[0] for k in munis]
    children: list = list(munis)
    parents: list = list(states)
    if depth == 3:
        for state in sorted(set(states)):
            children.append(state)
            parents.append(country_key)
    return pd.DataFrame({"parent": parents}, index=pd.Index(children, name="node", tupleize_cols=False))


def apply_nan_mask(da: xr.DataArray, *, fraction: float = 0.3, seed: int = 0) -> xr.DataArray:
    """Return a copy of `da` with `fraction` of spatial cells set to NaN.

    The masked cells are chosen by a seeded RNG over the (latitude, longitude)
    grid and shared across all batch dims (member/step), as a physical mask would
    be. This is **synthetic, for exercising the masked reduce path** - real 2t
    fields rarely carry true NaN - so it measures the renormalisation code path's
    cost rather than a physically meaningful mask.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    rng = np.random.default_rng(seed)
    n_lat = da.sizes["latitude"]
    n_lon = da.sizes["longitude"]
    keep = rng.random((n_lat, n_lon)) >= fraction  # True = keep, False = NaN out
    keep_da = xr.DataArray(keep, dims=("latitude", "longitude"))
    return da.where(keep_da)  # broadcasts over batch dims; returns a new array
