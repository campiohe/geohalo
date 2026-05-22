import shutil
import subprocess
from collections.abc import Iterator

import numpy as np
import pytest
import shapely

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec


@pytest.fixture
def simple_grid() -> GridSpec:
    """5x6 ascending-lat grid, 1° spacing, centered on the equator.

    Cell (i, j) is centered at (lat=lats[i], lon=lons[j]) and spans
    [lat - 0.5, lat + 0.5] x [lon - 0.5, lon + 0.5].
    """
    return GridSpec(
        lats=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        lons=np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
    )


@pytest.fixture
def descending_grid() -> GridSpec:
    """Same numeric content as `simple_grid` but built from descending lats."""
    return GridSpec(
        lats=np.array([2.0, 1.0, 0.0, -1.0, -2.0]),
        lons=np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
    )


@pytest.fixture
def high_latitude_grid() -> GridSpec:
    """5x6 grid around 70 °N — cell areas vary strongly with latitude here."""
    return GridSpec(
        lats=np.array([68.0, 69.0, 70.0, 71.0, 72.0]),
        lons=np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
    )


@pytest.fixture
def non_square_grid() -> GridSpec:
    """dlat != dlon — drives `resolve_factor`'s non-square-grid error."""
    return GridSpec(
        lats=np.array([0.0, 1.0, 2.0, 3.0]),
        lons=np.array([0.0, 2.0, 4.0, 6.0]),
    )


@pytest.fixture
def simple_polygons() -> PolygonSet:
    """Three polygons covering interesting cases against `simple_grid`.

    - sub_cell: entirely inside the (lat=0, lon=-0.5) cell ([-1, 0] x [-0.5, 0.5]).
    - multi_cell: a 2x2 box centered on the (lat=0.5, lon=0) cell corner.
    - irregular: an L-shaped polygon spanning several cells.
    """
    sub_cell = shapely.box(-0.9, -0.4, -0.1, 0.4)
    multi_cell = shapely.box(-0.3, -0.3, 0.7, 0.7)
    irregular = shapely.Polygon([
        (-2.0, -2.0), (-2.0, -1.0), (-1.0, -1.0), (-1.0, 0.0),
        (0.0, 0.0), (0.0, -2.0), (-2.0, -2.0),
    ])
    return PolygonSet.build(
        geoms=[sub_cell, multi_cell, irregular],
        keys=[("sub",), ("multi",), ("irregular",)],
        key_names=("name",),
    )


@pytest.fixture
def outside_polygons() -> PolygonSet:
    """A polygon far outside `simple_grid` for EmptyCoverageError tests."""
    return PolygonSet.build(
        geoms=[shapely.box(100.0, 100.0, 101.0, 101.0)],
        keys=[("away",)],
        key_names=("name",),
    )


@pytest.fixture
def high_latitude_polygons() -> PolygonSet:
    """Three polygons positioned to intersect `high_latitude_grid` (lat 68-72°N)."""
    sub_cell = shapely.box(-0.9, 69.6, -0.1, 70.4)
    multi_cell = shapely.box(-0.3, 69.7, 0.7, 70.7)
    irregular = shapely.Polygon([
        (-2.0, 68.0), (-2.0, 69.0), (-1.0, 69.0), (-1.0, 70.0),
        (0.0, 70.0), (0.0, 68.0), (-2.0, 68.0),
    ])
    return PolygonSet.build(
        geoms=[sub_cell, multi_cell, irregular],
        keys=[("sub",), ("multi",), ("irregular",)],
        key_names=("name",),
    )


def _docker_available() -> bool:
    """Probe whether `docker info` works."""
    docker_path = shutil.which("docker")
    if docker_path is None:
        return False
    try:
        result = subprocess.run(
            [docker_path, "info"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


@pytest.fixture(scope="session")
def redis_container() -> Iterator[object]:
    from testcontainers.redis import RedisContainer  # noqa: PLC0415

    with RedisContainer() as container:
        yield container


@pytest.fixture
def redis_client(redis_container: object) -> Iterator[object]:
    import redis  # noqa: PLC0415

    host = redis_container.get_container_host_ip()
    port = int(redis_container.get_exposed_port(6379))
    client = redis.Redis(host=host, port=port)
    client.flushdb()
    yield client
    client.flushdb()


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Auto-skip @pytest.mark.redis tests when Docker is unreachable."""
    if _docker_available():
        return
    skip = pytest.mark.skip(reason="docker unavailable; skipping redis-marked tests")
    for item in items:
        if "redis" in item.keywords:
            item.add_marker(skip)
