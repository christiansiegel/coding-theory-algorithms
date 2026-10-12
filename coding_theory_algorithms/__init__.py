"""Public classes and the shared binary field for textbook coding exercises."""

from importlib import import_module


__all__ = ["BSC", "GaloisField", "GF2", "LinearBlockCode", "CyclicCode", "BCHCode", "RSCode"]

_EXPORT_MODULES = {
    "BSC": "binary_symmetric_channel",
    "GaloisField": "galois_field",
    "GF2": "galois_field",
    "LinearBlockCode": "linear_block_code",
    "CyclicCode": "cyclic_code",
    "BCHCode": "bch_code",
    "RSCode": "rs_code",
}


def __getattr__(name):
    # Lazy exports keep python -m submodule demonstrations free of runpy warnings.
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
