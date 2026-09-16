"""Public byte-serialization methods shared by precomputed objects."""

from typing import Self


class NPZSerializable:
    def to_npz(self) -> bytes:
        """Return a versioned NPZ archive of numeric arrays and UTF-8 JSON metadata.

        No object arrays or executable Python objects are stored. Supported key
        types retain their values, names, dtypes, and order; unsupported custom
        labels raise TypeError. Write these bytes to a file or your own store.
        """
        from geohalo._npz import encode  # noqa: PLC0415 - avoid class import cycles

        return encode(self)

    @classmethod
    def from_npz(cls, blob: bytes) -> Self:
        """Restore this type from NPZ bytes, always using allow_pickle=False.

        Wrong object types, unknown schema versions, and malformed payloads
        raise ValueError. Legacy cache formats are never loaded. This validates
        structure, not provenance: authenticate artifacts and bound their size
        separately when accepting them from untrusted storage.
        """
        from geohalo._npz import decode  # noqa: PLC0415 - avoid class import cycles

        return decode(blob, cls)
