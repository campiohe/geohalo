from collections.abc import Callable

import pytest
import shapely


@pytest.fixture
def box() -> Callable[[int], shapely.Polygon]:
    """Factory: `box(i)` returns a unit square anchored at (i, i)."""
    def _box(i: int) -> shapely.Polygon:
        return shapely.box(i, i, i + 1, i + 1)
    return _box
