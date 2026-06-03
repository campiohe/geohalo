"""BiasTree: parent-child rollup operator."""

import hashlib
import numbers
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class BiasTree:
    rollup_matrix: sp.csr_matrix
    keys: pd.Index
    digest: bytes
    how: str = "mean"

    @property
    def leaf_keys(self) -> pd.Index:
        _, n_leaves = self.rollup_matrix.shape
        return self.keys[:n_leaves]

    def __repr__(self) -> str:
        return (
            f"BiasTree(nodes={len(self.keys)}, leaves={self.rollup_matrix.shape[1]}, "
            f"nnz={self.rollup_matrix.nnz})"
        )

    @classmethod
    def compute(
        cls,
        edges: pd.DataFrame,
        *,
        parent_col: str = "parent",
        weight_col: str | None = None,
        how: Literal["mean", "sum"] = "mean",
    ) -> "BiasTree":
        if not isinstance(edges, pd.DataFrame):
            raise TypeError(f"edges must be a pd.DataFrame, got {type(edges).__name__}")
        if not edges.index.is_unique:
            raise ValueError("edges.index has duplicates; expected one row per child (tree, not DAG)")
        if parent_col not in edges.columns:
            raise ValueError(f"parent_col {parent_col!r} not in {list(edges.columns)}")

        children = list(edges.index)
        parents = list(edges[parent_col])
        weights = list(edges[weight_col]) if weight_col is not None else [1.0] * len(children)
        for w in weights:
            # numbers.Real covers Python and numpy int/float scalars (np.int64 is *not* an `int`).
            if not (isinstance(w, numbers.Real) and np.isfinite(w) and w > 0):
                raise ValueError(f"edge weight must be positive finite, got {w!r}")

        parent_of: dict[Hashable, Hashable] = dict(zip(children, parents, strict=True))
        children_of: dict[Hashable, list[tuple[Hashable, float]]] = {}
        for c, p, w in zip(children, parents, weights, strict=True):
            children_of.setdefault(p, []).append((c, float(w)))

        all_nodes = set(children) | set(parents)
        leaves = {n for n in all_nodes if n not in children_of}
        if not leaves:
            raise ValueError("edges must have at least one leaf; got a pure cycle or empty edges")
        depth = _node_depth(parent_of, leaves)
        if len(depth) != len(all_nodes):
            unreached = sorted(all_nodes - set(depth), key=repr)[:5]
            raise ValueError(f"cycle detected: nodes unreachable from any leaf: {unreached!r}")

        sorted_leaves = sorted(leaves, key=repr)
        internals = sorted((n for n in all_nodes if n not in leaves), key=lambda n: (depth[n], repr(n)))
        nodes_list = sorted_leaves + internals
        node_index = {n: i for i, n in enumerate(nodes_list)}
        leaf_index = {n: i for i, n in enumerate(sorted_leaves)}
        n_leaves = len(sorted_leaves)

        matrix = sp.lil_matrix((len(nodes_list), n_leaves), dtype=np.float64)
        for leaf, j in leaf_index.items():
            matrix[node_index[leaf], j] = 1.0
        for node in internals:
            entries = children_of[node]
            total = sum(w for _, w in entries)
            composed = sp.csr_matrix((1, n_leaves), dtype=np.float64)
            for child, w in entries:
                scale = (w / total) if how == "mean" else w
                composed = composed + matrix.getrow(node_index[child]) * scale
            matrix[node_index[node]] = composed

        return cls(
            rollup_matrix=matrix.tocsr(),
            keys=_build_keys(nodes_list, edges.index),
            digest=tree_digest(edges, parent_col=parent_col, weight_col=weight_col, how=how),
            how=how,
        )


def _build_keys(nodes_list: list[Hashable], index: pd.Index) -> pd.Index:
    """Build the node-keys Index, matching the input MultiIndex when arity allows.

    When the input is a ``pd.MultiIndex`` and *every* node — each leaf and each
    parent — is a tuple of ``index.nlevels`` levels (a same-arity rollup, e.g.
    ``(scenario, region)`` leaves into ``(scenario, "ALL")``), the output is a
    real ``pd.MultiIndex`` with the level names preserved, so the rolled-up
    ``geom`` coord stays selectable on its levels like every other operator.

    A varying-arity hierarchy (parents shorter than leaves, e.g.
    ``(BR, SP, muni)`` into ``(BR, SP)`` into ``(BR,)``) cannot be one
    MultiIndex — pandas requires a uniform level count — so it falls back to a
    flat object Index of tuples, as do scalar keys. ``tupleize_cols=False`` stops
    pandas auto-tupleizing the tuple nodes back into a (NaN-padded) MultiIndex.
    """
    if isinstance(index, pd.MultiIndex):
        nlevels = index.nlevels
        same_arity = all(isinstance(n, tuple) and len(n) == nlevels for n in nodes_list)
        if same_arity:
            return pd.MultiIndex.from_tuples(nodes_list, names=index.names)
    return pd.Index(nodes_list, name=index.name, tupleize_cols=False)


def _node_depth(parent_of: dict[Hashable, Hashable], leaves: set[Hashable]) -> dict[Hashable, int]:
    depth = dict.fromkeys(leaves, 0)
    for leaf in leaves:
        cur, d, seen = leaf, 0, {leaf}
        while cur in parent_of:
            parent = parent_of[cur]
            if parent in seen:
                raise ValueError(f"cycle detected at node {parent!r}")
            seen.add(parent)
            d += 1
            depth[parent] = max(depth.get(parent, 0), d)
            cur = parent
    return depth


def tree_digest(
    edges: pd.DataFrame,
    *,
    parent_col: str = "parent",
    weight_col: str | None = None,
    how: str = "mean",
) -> bytes:
    """Cache key for a bias tree, derivable from inputs without building it."""
    h = hashlib.sha256()
    h.update(how.encode())
    # A MultiIndex has no scalar `.name` (it would be None, dropping the level names);
    # hash its `.names` list so trees differing only in level names get distinct keys.
    # Flat indices keep hashing the scalar name, so their digests are unchanged.
    if isinstance(edges.index, pd.MultiIndex):
        h.update(repr(list(edges.index.names)).encode())
    else:
        h.update(repr(edges.index.name).encode())
    if weight_col is not None:
        rows = list(zip(edges.index, edges[parent_col], edges[weight_col], strict=True))
    else:
        rows = list(zip(edges.index, edges[parent_col], strict=True))
    rows.sort(key=lambda r: tuple(repr(x) for x in r))
    for row in rows:
        for item in row:
            h.update(repr(item).encode())
    return h.digest()
