from collections import Counter

import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore, WrapperStore

from geohalo import RestrictedOperator, reduce_with_operator, reduce_with_restricted_operator
from tests.restricted_operator._helpers import make_grid, make_operator


class CountingStore(WrapperStore):
    """Count actual Zarr store reads, including the synchronous fast path."""

    def __init__(self, store, reads=None):
        super().__init__(store)
        self.reads = [] if reads is None else reads

    def _with_store(self, store):
        return type(self)(store, self.reads)

    async def get(self, key, prototype, byte_range=None):
        self.reads.append(key)
        return await super().get(key, prototype, byte_range)

    def get_sync(self, key, *, prototype=None, byte_range=None):
        self.reads.append(key)
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    async def _get_many(self, requests):
        for key, prototype, byte_range in requests:
            yield key, await self.get(key, prototype, byte_range)


@pytest.mark.parametrize("dask", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_zarr_v3_reads_only_touched_chunks_once(dask, descending, transpose, skipna):
    operator = make_operator()
    eager = make_grid(operator, descending=descending, batch_shape=(5,)).drop_vars("model")
    if skipna:
        eager.values[..., 7 if descending else 0, 1] = np.nan
    store = CountingStore(MemoryStore())
    eager.to_dataset().to_zarr(
        store, encoding={"t2m": {"chunks": (2, 2, 2)}}, zarr_format=3, consolidated=False,
    )
    with xr.open_zarr(store, chunks={} if dask else None, consolidated=False) as ds:
        store.reads.clear()
        lazy = ds.t2m.transpose("longitude", "dim0", "latitude") if transpose else ds.t2m
        plan = RestrictedOperator.from_grid(operator, lazy)
        assert not any(key.startswith("t2m/c/") for key in store.reads)
        got = reduce_with_restricted_operator(lazy, plan, skipna=skipna)
        xr.testing.assert_identical(got, reduce_with_operator(eager, operator, skipna=skipna))
        rows, cols = np.divmod(operator.matrix.indices, 10)
        if descending:
            rows = 7 - rows
        touched = set(zip(rows // 2, cols // 2, strict=True))
        expected = Counter(f"t2m/c/{t}/{r}/{c}" for t in range(3) for r, c in touched)
        actual = Counter(key for key in store.reads if key.startswith("t2m/c/"))
        assert actual == expected
        assert len(actual) < 3 * 4 * 5
