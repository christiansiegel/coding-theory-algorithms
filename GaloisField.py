"""Compatibility entry point for coding_theory_algorithms.galois_field."""

from importlib import import_module
import sys

if __name__ == "__main__":
    from runpy import run_module

    run_module("coding_theory_algorithms.galois_field", run_name="__main__")
else:
    # Share the module object, including singleton fields and patched attributes.
    sys.modules[__name__] = import_module("coding_theory_algorithms.galois_field")
