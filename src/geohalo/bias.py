import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr


def _topological_levels(
    edges: list[tuple[tuple, tuple, float]],
) -> tuple[list[tuple], list[tuple], list[tuple], dict[tuple, list[tuple[tuple, float]]]]:
    """Return (sorted_nodes, topo_order, leaf_keys, parent_to_children).

    `sorted_nodes` is the user-facing order (deterministic key sort).
    `topo_order` is leaves-first then parents-whose-children-are-resolved;
    matrix composition must iterate it so each parent sees fully-populated
    child rows. Raises ValueError on any cycle, with one offending node
    in the message.
    """
    parent_to_children: dict[tuple, list[tuple[tuple, float]]] = {}
    all_nodes: set[tuple] = set()
    parents: set[tuple] = set()
    in_degree: dict[tuple, int] = {}  # # of children below this node still unprocessed
    for parent, child, weight in edges:
        parent_to_children.setdefault(parent, []).append((child, weight))
        all_nodes.add(parent)
        all_nodes.add(child)
        parents.add(parent)
        in_degree[parent] = in_degree.get(parent, 0) + 1

    # Kahn's algorithm processed from leaves upward.
    leaf_keys = sorted(n for n in all_nodes if n not in parents)
    ready: list[tuple] = sorted(n for n in all_nodes if in_degree.get(n, 0) == 0)

    # Reverse lookup: child -> parents (for in_degree decrement as we process).
    child_to_parents: dict[tuple, list[tuple]] = {}
    for parent, child, _ in edges:
        child_to_parents.setdefault(child, []).append(parent)

    topo_order: list[tuple] = []
    while ready:
        n = ready.pop(0)
        topo_order.append(n)
        for parent in child_to_parents.get(n, ()):
            in_degree[parent] -= 1
            if in_degree[parent] == 0:
                bisect.insort(ready, parent)

    if len(topo_order) != len(all_nodes):
        unprocessed = sorted(all_nodes - set(topo_order))
        raise ValueError(
            f"hierarchy contains a cycle; nodes left unprocessed: {unprocessed[:5]!r}"
            + (" ..." if len(unprocessed) > 5 else ""),
        )

    return sorted(all_nodes), topo_order, leaf_keys, parent_to_children


def _compose_matrix(
    nodes: list[tuple],
    topo_order: list[tuple],
    leaf_keys: list[tuple],
    parent_to_children: dict[tuple, list[tuple[tuple, float]]],
) -> sp.csr_matrix:
    """Build the (n_nodes, n_leaves) sparse matrix M.

    Leaf rows are identity (M[leaf_i, leaf_col(leaf_i)] = 1).
    Internal rows are the normalized-weighted sum of child rows; iterating
    in topological order guarantees every child row is populated by the
    time we process its parent (critical for multi-level hierarchies
    where one internal is the child of another).
    """
    n_nodes = len(nodes)
    n_leaves = len(leaf_keys)
    node_index = {n: i for i, n in enumerate(nodes)}
    leaf_index = {n: j for j, n in enumerate(leaf_keys)}
    leaf_set = set(leaf_keys)

    # The composed matrix is dense per-row during construction but ends up
    # sparse overall. Build with LIL for cheap row assembly, finalize to CSR.
    matrix = sp.lil_matrix((n_nodes, n_leaves), dtype=np.float64)
    for node in topo_order:
        if node in leaf_set:
            matrix[node_index[node], leaf_index[node]] = 1.0
            continue
        children = parent_to_children[node]
        total = sum(w for _, w in children)
        row_idx = node_index[node]
        composed = sp.csr_matrix((1, n_leaves), dtype=np.float64)
        for child, w in children:
            composed = composed + matrix.getrow(node_index[child]) * (w / total)
        matrix[row_idx] = composed
    return matrix.tocsr()


def _compute_digest(
    edges: list[tuple[tuple, tuple, float]],
    key_names: tuple[str, ...],
) -> bytes:
    h = hashlib.sha256()
    h.update(repr(key_names).encode())
    for parent, child, weight in sorted(edges):
        h.update(repr(parent).encode())
        h.update(repr(child).encode())
        h.update(repr(float(weight)).encode())
    return h.digest()


@dataclass(frozen=True, repr=False)
class BiasHierarchy:
    nodes: list[tuple]
    key_names: tuple[str, ...]
    leaf_keys: list[tuple]
    matrix: sp.csr_matrix
    digest: bytes

    def __repr__(self) -> str:
        return (
            f"BiasHierarchy(nodes={len(self.nodes)}, leaves={len(self.leaf_keys)}, "
            f"key_names={self.key_names}, digest={self.digest[:4].hex()}...)"
        )

    @classmethod
    def build(
        cls,
        edges: list[tuple[tuple, tuple, float]],
        *,
        key_names: tuple[str, ...] = ("polygon_id",),
    ) -> "BiasHierarchy":
        if not edges:
            raise ValueError("BiasHierarchy must be built from a non-empty edge list")
        if not key_names:
            raise ValueError("key_names must contain at least one name")
        arity = len(key_names)
        seen_edges: set[tuple[tuple, tuple]] = set()
        for edge in edges:
            parent, child, weight = edge
            for label, key in (("parent", parent), ("child", child)):
                if not isinstance(key, tuple):
                    raise TypeError(
                        f"each {label} key must be a tuple, got {type(key).__name__}: {key!r}",
                    )
                if len(key) != arity:
                    raise ValueError(
                        f"{label} key {key!r} has length {len(key)}, expected {arity} to match key_names={key_names}",
                    )
            if not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"edge weight must be positive finite, got {weight!r} on edge ({parent!r} -> {child!r})",
                )
            pair = (parent, child)
            if pair in seen_edges:
                raise ValueError(f"duplicate edge: parent={parent!r}, child={child!r}")
            seen_edges.add(pair)

        all_nodes, topo_order, leaf_keys, parent_to_children = _topological_levels(edges)
        matrix = _compose_matrix(all_nodes, topo_order, leaf_keys, parent_to_children)
        digest = _compute_digest(edges, key_names)

        return cls(
            nodes=all_nodes,
            key_names=key_names,
            leaf_keys=leaf_keys,
            matrix=matrix,
            digest=digest,
        )


SUPPORTED_NAN_MODES: tuple[str, ...] = ("ignore", "raise")


def compute_bias(
    aggregated: xr.DataArray,
    hierarchy: BiasHierarchy,
    *,
    polygon_dim: str = "polygon",
    on_nan_child: Literal["ignore", "raise"] = "ignore",
) -> xr.DataArray:
    """Roll a per-leaf aggregate up a parent-child-weight DAG.

    Performs one sparse · dense matmul (two for the NaN-aware path) using
    the pre-composed `hierarchy.matrix`. Leaves pass through unchanged;
    each internal node is the renormalized weighted average of its
    transitively-contributing leaves.

    on_nan_child:
      - "ignore": NaN-aware leaf-level renormalization. A parent's value
        is the weighted average of its non-NaN contributing leaves; a
        parent with no finite contributing leaf is NaN.
      - "raise": any NaN leaf contributing to a parent raises
        ValueError with the offending node keys.
    """
    if on_nan_child not in SUPPORTED_NAN_MODES:
        raise ValueError(
            f"on_nan_child must be one of {SUPPORTED_NAN_MODES}, got {on_nan_child!r}",
        )

    if polygon_dim not in aggregated.dims:
        raise ValueError(f"aggregated has no dim named {polygon_dim!r}; dims={aggregated.dims}")
    poly_index = aggregated[polygon_dim].to_index()
    index_names = tuple(poly_index.names) if poly_index.names is not None else ()
    if index_names != hierarchy.key_names:
        raise ValueError(
            f"polygon index key_names {index_names!r} do not match hierarchy.key_names {hierarchy.key_names!r}",
        )

    available = set(poly_index)
    missing = [k for k in hierarchy.leaf_keys if k not in available]
    if missing:
        raise ValueError(
            f"missing leaf(s) in aggregated input: {missing[:5]!r}" + (" ..." if len(missing) > 5 else ""),
        )

    leaves_da = aggregated.sel({polygon_dim: hierarchy.leaf_keys})
    batch_dims = [d for d in leaves_da.dims if d != polygon_dim]
    leaves_arr = leaves_da.transpose(*batch_dims, polygon_dim).to_numpy()
    batch_shape = leaves_arr.shape[:-1]
    leaves_flat = leaves_arr.reshape(-1, len(hierarchy.leaf_keys))

    is_nan = np.isnan(leaves_flat)
    if on_nan_child == "raise" and is_nan.any():
        m_bool = hierarchy.matrix.astype(bool).astype(np.float64)
        contributes_nan = (m_bool @ is_nan.T).T
        bad_node_idx = np.flatnonzero(contributes_nan.any(axis=0))
        leaf_set = set(hierarchy.leaf_keys)
        bad_keys = [hierarchy.nodes[i] for i in bad_node_idx if hierarchy.nodes[i] not in leaf_set]
        if bad_keys:
            raise ValueError(
                f"on_nan_child='raise': NaN leaf contributes to node(s) {bad_keys[:5]!r}"
                + (" ..." if len(bad_keys) > 5 else ""),
            )

    if not is_nan.any():
        out_flat = (hierarchy.matrix @ leaves_flat.T).T
    else:
        filled = np.where(is_nan, 0.0, leaves_flat)
        mask = (~is_nan).astype(np.float64)
        numer = (hierarchy.matrix @ filled.T).T
        denom = (hierarchy.matrix @ mask.T).T
        with np.errstate(invalid="ignore", divide="ignore"):
            out_flat = np.where(denom > 0, numer / denom, np.nan)

    out = out_flat.reshape(*batch_shape, len(hierarchy.nodes))
    out_index = pd.MultiIndex.from_tuples(hierarchy.nodes, names=list(hierarchy.key_names))
    return xr.DataArray(
        out,
        dims=(*batch_dims, polygon_dim),
        coords={
            **{d: leaves_da[d] for d in batch_dims},
            polygon_dim: out_index,
        },
    )
