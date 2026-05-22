"""RedisWeightCache constructor raises ImportError if `redis` isn't installed."""

from unittest import mock

import pytest

from geohalo import cache as cache_module
from geohalo.cache import RedisWeightCache


def test_construct_without_redis_module_raises() -> None:
    """Monkeypatch the module-level error flag to simulate missing redis."""
    with (
        mock.patch.object(cache_module, "_REDIS_IMPORT_ERROR", ImportError("no redis")),
        pytest.raises(ImportError, match="requires the 'redis' package"),
    ):
        RedisWeightCache(client=object())  # type: ignore[arg-type]
