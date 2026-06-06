# Floored mean-preserving downscaling (`floor=` on `resample_grid`)

**Date:** 2026-06-06
**Status:** Approved design, pre-implementation

## Problem

The mean-preserving downscale `y = y_op + P(x − A·y_op)` can produce negative
values from non-negative input (precipitation, wind speed, concentrations):

- The refinement series `y_op = (Σ Gʲ)·B·x`, `G = I − B·A`, adds signed
  correction terms that overshoot near sharp gradients (a wet cell next to dry
  cells), even though `B` alone is a convex combination.
- The final correction `P(x − A·y_op)` shifts each source-cell block by a
  constant; a block whose bilinear surface overestimated the parent mean gets
  shifted below zero.

Even `iterations=1` can push a dry cell adjacent to a wet cell negative.

## Goal

Optional lower bound on `resample_grid` output that keeps **exact per-block
mean preservation** (each source cell's children still average to the source
value) while guaranteeing `output ≥ floor`.

## Non-goals

- **No `floor` on `reduce(..., target_resolution=...)`.** Clamping is
  nonlinear and value-dependent; it cannot pass through `fuse_left`, which the
  fused `ReduceOperator` hot path depends on. Users who need it do the explicit
  two-step `resample_grid(floor=...)` → `reduce`; the docs say so.
- **No upper bound / two-sided clamp.** One-sided rescale is exact in a single
  pass; two-sided needs iterative redistribution. Out of scope.
- **No change to the linear operator.** `Resampler.transform_matrix`,
  `resampler_digest`, the cache payloads, and `PAYLOAD_VERSION` are untouched.
  The floor is a value-dependent post-step at apply time only.

## Chosen approach: clip & multiplicative block rescale at apply time

After the existing linear apply `y = T·x`, per source-cell block (the children
sharing a nearest parent — the same blocks `P`/`A` define):

```
y' = max(y, floor)
s  = (x_parent − floor) / mean(y' − floor)        # one scale per block
y″ = floor + (y' − floor) · s
```

`y' − floor ≥ 0` and `x_parent − floor ≥ 0` give `s ≥ 0`, so `y″ ≥ floor`, and
`mean(y″) = floor + s·mean(y' − floor) = x_parent` — the floor holds **and**
the block mean is restored exactly, in one closed-form pass.

Mean preservation here means the **uniform (unweighted) arithmetic mean over
each block's children**, matching what `A` already preserves today.

### Block rules, in precedence order

1. **NaN blocks stay NaN.** Any block that is NaN under current behaviour
   remains NaN; the NaN footprint does not change.
2. **Source below floor → block filled with `floor`.** Floor is guaranteed
   everywhere in the output; the mean is knowingly broken only for blocks
   whose source already violated the floor (treated as input cleanup, and
   documented).
3. **All children clipped to floor but parent above floor → constant fill with
   the parent value** (`s` would be 0/0; the constant fill preserves the mean
   and the floor trivially).
4. **Otherwise → clip and rescale** per the formula above.

Rescale denominators are guarded: rule 4 only applies where
`mean(y' − floor) > 0` and finite; rules 2–3 catch the rest.

## API

```python
def resample_grid(
    source, target_resolution, *, lat_dim="latitude", lon_dim="longitude",
    iterations=1, floor: float | Mapping[str, float] | None = None,
) -> T

def resample_grid_with_matrix(
    source, resampler, *, lat_dim="latitude", lon_dim="longitude",
    floor: float | Mapping[str, float] | None = None,
) -> T
```

- `floor=None` (default): current behaviour, **bit-for-bit identical output**.
- `floor=<float>`: DataArray → floored; Dataset → every spatial var floored.
- `floor=<Mapping[str, float]>`: Dataset only — named spatial vars floored
  (each at its own value), unnamed vars pass through linear.
  - Mapping with a DataArray input → `ValueError` (strict; no `da.name`
    matching).
  - Mapping naming a variable not among the Dataset's **spatial** data vars →
    `ValueError` (catches typos).

## Components

### `geometry.floor_blocks` (new pure function, `geometry.py`)

```python
def floor_blocks(
    resampled: np.ndarray,   # (batch, n_target) — output of flat @ T.T
    source_flat: np.ndarray, # (batch, n_source) — the same flat input
    parent_flat: np.ndarray, # (n_target,) int — nearest source cell per target cell
    floor: float,
) -> np.ndarray              # (batch, n_target)
```

Pure NumPy/Scipy, no operator state — consistent with `geometry.py`'s charter.
Per-block means via bincount-style accumulation (or a small CSR averaging
matrix built once per call from `parent_flat`); vectorised over the batch axis.
Implements block rules 1–4.

### Wiring (`api.py`)

- `_apply_matrix_da` already has both the flattened source (`flat`) and the
  flattened output (`out_flat`) in hand — the post-step slots in between matmul
  and reshape, behind `if floor is not None`.
- `parent_flat` is recomputed at apply time from
  `nearest_index(lat) ⊗ nearest_index(lon)` (two 1-D searchsorted calls — the
  same recipe `_build_factors` uses). No new field on `Resampler`, hence no
  digest, serialization, or `PAYLOAD_VERSION` change.
- Dataset path: `_map_spatial_vars` resolves the per-var floor (float → same
  value for all; mapping → lookup or `None`) before delegating to the
  DataArray path. Mapping validation happens before any work.

### Implementation caution: index ordering

`parent_flat` indexes **source** cells and is consumed against the flattened
layouts inside `_apply_matrix_da`, which canonicalises a descending latitude
axis ascending (`sortby`) before flattening. The `parent_flat` computation must
use the same orderings as the data actually applied — verify against the
existing descending-lat (ECMWF) tests, not just ascending grids.

## Docs

- `floor` documented in both function docstrings.
- Short subsection on the mean-preserving-downscaling concepts page
  (`docs/.../downscaling`): what the floor guarantees, the exact-mean property,
  and the rule-2 caveat (blocks whose source already violates the floor are
  filled with the floor — flagged as input cleanup).
- The `reduce` docs gain one line pointing floor-needing users at the explicit
  `resample_grid(floor=...) → reduce` two-step.

## Testing

All in the existing pytest style (warnings-as-errors, ≥85 % coverage gate):

- **Floor holds:** `output.min() ≥ floor` on adversarial fields — sharp
  wet/dry boundaries, high `iterations`.
- **Mean preserved:** per-block child means equal source values within float
  tolerance, with and without flooring (excluding rule-2 blocks).
- **`floor=None` is a no-op:** arrays identical to current behaviour.
- **Block rules:** a direct test each for rules 1 (NaN footprint unchanged),
  2 (source below floor → floor fill), 3 (all-clipped block → parent fill).
- **Dataset semantics:** float floors all spatial vars; mapping floors only
  named vars; mapping + DataArray raises; unknown var name raises.
- **Batched:** extra dims (time/ensemble) — each slice matches the unbatched
  result.
- **Descending lats:** ECMWF-style grid produces the same floored output as
  its ascending twin.
- **Hypothesis property test:** random non-negative fields → output
  non-negative and per-block means preserved.
