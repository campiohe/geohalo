"""Shared fixtures: GADM fetch, polygon construction, grid, hierarchies.

Used by `benchmarks/run.py` and `examples/visualize.py`.
"""

import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely

from geohalo.bias import BiasHierarchy
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BRA_2.json.zip"

# 0.25° resolution over a 40°×40° Brazil bbox -> 160×160 = 25,600 cells.
BRAZIL_BBOX = (-74.0, -34.0, -34.0, 6.0)  # xmin, ymin, xmax, ymax
BRAZIL_RESOLUTION_DEG = 0.25

_SUBSET_LABELS = ("small", "medium", "large")


def fetch_gadm_brazil_l2(cache_dir: Path) -> dict:
    """Download (and cache) GADM 4.1 Brazil level-2 GeoJSON.

    Returns the parsed GeoJSON dict. ~6 MB compressed, ~80 MB uncompressed.
    """
    json_path = cache_dir / "gadm41_BRA_2.json"
    if not json_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "gadm41_BRA_2.json.zip"
        print(f"downloading GADM Brazil L2 to {zip_path} ...")
        with urllib.request.urlopen(GADM_URL, timeout=120) as resp:  # noqa: S310
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".json"))
            json_path.write_bytes(zf.read(inner))
        zip_path.unlink()
    return json.loads(json_path.read_text())


def build_polygon_set(
    geojson: dict, bbox: tuple[float, float, float, float],
) -> tuple[PolygonSet, dict[tuple, tuple[str, str]]]:
    """Build a PolygonSet from GADM features filtered to a bbox.

    Returns the PolygonSet (keyed by (GID_1, GID_2)) plus a name lookup
    {key: (state_name, municipality_name)} useful for human-readable titles.
    """
    bbox_geom = shapely.box(*bbox)
    keys: list[tuple] = []
    geoms: list[shapely.Geometry] = []
    names: dict[tuple, tuple[str, str]] = {}
    for feature in geojson["features"]:
        geom = shapely.geometry.shape(feature["geometry"])
        if not geom.intersects(bbox_geom):
            continue
        props = feature["properties"]
        key = (props["GID_1"], props["GID_2"])
        keys.append(key)
        geoms.append(geom)
        names[key] = (props.get("NAME_1", ""), props.get("NAME_2", ""))
    if not keys:
        raise RuntimeError(f"no GADM polygons intersect bbox {bbox!r}")
    return (
        PolygonSet.build(geoms=geoms, keys=keys, key_names=("gid_1", "gid_2")),
        names,
    )


def make_brazil_grid() -> GridSpec:
    """Synthetic 0.25° lat/lon grid over the Brazil bbox.

    Cell centers are placed at bbox-min + 0.5*dx steps so cells fully tile
    the bbox without spilling over.
    """
    xmin, ymin, xmax, ymax = BRAZIL_BBOX
    d = BRAZIL_RESOLUTION_DEG
    n_lat = int(round((ymax - ymin) / d))
    n_lon = int(round((xmax - xmin) / d))
    lats = ymin + d * (np.arange(n_lat) + 0.5)
    lons = xmin + d * (np.arange(n_lon) + 0.5)
    return GridSpec(lats=lats, lons=lons)


def sample_polygons(polygons: PolygonSet, label: str) -> PolygonSet:
    """Deterministic subset of `polygons` by label.

    - "small": first 50 polygons in sorted-key order
    - "medium": every 11th polygon (modulo 11)
    - "large": the input set unchanged
    """
    if label not in _SUBSET_LABELS:
        raise ValueError(f"label must be one of {_SUBSET_LABELS}, got {label!r}")
    if label == "large":
        return polygons
    if label == "small":
        indices = list(range(min(50, len(polygons.keys))))
    else:  # medium
        indices = list(range(0, len(polygons.keys), 11))
    keys = [polygons.keys[i] for i in indices]
    geoms = [polygons.geoms[i] for i in indices]
    return PolygonSet.build(geoms=geoms, keys=keys, key_names=polygons.key_names)


def build_gadm_hierarchy(polygons: PolygonSet) -> BiasHierarchy:
    """Depth-2 GADM hierarchy: state (gid_1) -> municipality (gid_1, gid_2).

    All leaves keep their natural (gid_1, gid_2) key; state-level parents are
    keyed as (gid_1, "__state__") to share the (gid_1, gid_2) arity of the
    underlying PolygonSet (BiasHierarchy requires every node to have the same
    key arity). Edge weights are uniformly 1.0.
    """
    edges: list[tuple[tuple, tuple, float]] = []
    for gid_1, gid_2 in polygons.keys:
        state_key = (gid_1, "__state__")
        leaf_key = (gid_1, gid_2)
        edges.append((state_key, leaf_key, 1.0))
    return BiasHierarchy.build(edges, key_names=("gid_1", "gid_2"))


def build_deep_hierarchy(
    n_leaves: int,
    *,
    n_macros: int = 5,
    n_states_per_macro: int = 6,
    n_mesos_per_state: int = 4,
) -> BiasHierarchy:
    """Synthetic 4-level hierarchy: macro -> state -> meso -> leaf.

    Generates `n_leaves` integer-keyed leaves distributed round-robin across
    `n_macros * n_states_per_macro * n_mesos_per_state` meso-regions.
    Edge weights uniformly 1.0.

    Used purely for benchmarking compute_bias on a deeper DAG; not tied to
    any real polygon set.
    """
    if n_leaves < 1:
        raise ValueError(f"n_leaves must be >= 1, got {n_leaves}")

    edges: list[tuple[tuple, tuple, float]] = []
    triples: list[tuple[int, int, int]] = []
    for macro_i in range(n_macros):
        for state_j in range(n_states_per_macro):
            state_id = macro_i * n_states_per_macro + state_j
            for meso_k in range(n_mesos_per_state):
                triples.append((macro_i, state_id, meso_k))

    leaves_per_meso: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for leaf_i in range(n_leaves):
        triples_idx = leaf_i % len(triples)
        leaves_per_meso[triples[triples_idx]].append(leaf_i)

    seen_meso: set[tuple[int, int, int]] = set()
    seen_state: set[tuple[int, int]] = set()
    for (macro_i, state_id, meso_k), leaf_ids in leaves_per_meso.items():
        macro_key = (macro_i, -1, -1, -1)
        state_key = (macro_i, state_id, -1, -1)
        meso_key = (macro_i, state_id, meso_k, -1)
        if (macro_i, state_id) not in seen_state:
            edges.append((macro_key, state_key, 1.0))
            seen_state.add((macro_i, state_id))
        if (macro_i, state_id, meso_k) not in seen_meso:
            edges.append((state_key, meso_key, 1.0))
            seen_meso.add((macro_i, state_id, meso_k))
        for leaf_i in leaf_ids:
            leaf_key = (macro_i, state_id, meso_k, leaf_i)
            edges.append((meso_key, leaf_key, 1.0))

    return BiasHierarchy.build(
        edges,
        key_names=("macro", "state", "meso", "leaf"),
    )
