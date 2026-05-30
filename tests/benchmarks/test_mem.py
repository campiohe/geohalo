import pytest

from benchmarks._mem import gb_to_bytes, run_guarded


def test_gb_to_bytes_uses_binary_gigabytes():
    assert gb_to_bytes(1) == 1024 ** 3
    assert gb_to_bytes(10) == 10 * 1024 ** 3


def test_gb_to_bytes_accepts_fractional():
    assert gb_to_bytes(0.5) == 1024 ** 3 // 2


def test_run_guarded_returns_value_and_no_error_on_success():
    result, error = run_guarded(lambda: 42)
    assert result == 42
    assert error is None


def test_run_guarded_catches_memory_error():
    def boom():
        raise MemoryError("simulated")

    result, error = run_guarded(boom)
    assert result is None
    assert error == "skipped"


def test_run_guarded_propagates_other_errors():
    def boom():
        raise ValueError("not memory")

    with pytest.raises(ValueError, match="not memory"):
        run_guarded(boom)
