import shutil
import subprocess
from collections.abc import Iterator

import pytest


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
