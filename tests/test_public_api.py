import geohalo


def test_public_exports() -> None:
    for name in ("reduce", "reduce_with_stencil", "resample_grid",
                 "resample_grid_with_matrix", "aggregate_bias",
                 "aggregate_bias_with_tree", "Stencil", "Resampler",
                 "BiasTree", "LocalCache", "RedisCache", "EmptyOverlapError"):
        assert hasattr(geohalo, name), f"missing export: {name}"


def test_no_legacy_exports() -> None:
    for name in ("Weights", "PolygonSet", "GridSpec", "BiasHierarchy",
                 "compute_weights", "compute_bias", "LocalWeightCache",
                 "RedisWeightCache", "aggregate"):
        assert not hasattr(geohalo, name), f"unexpected legacy export: {name}"
