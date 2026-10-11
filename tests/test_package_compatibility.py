"""Protect imports, shared field objects, and script entry points after moving files."""

import pytest

from .test_smoke import run_python


@pytest.mark.parametrize(
    "legacy,modern,name",
    [
        ("GaloisField", "galois_field", "GaloisField"),
        ("LinearBlockCode", "linear_block_code", "LinearBlockCode"),
        ("CyclicCode", "cyclic_code", "CyclicCode"),
        ("BCHCode", "bch_code", "BCHCode"),
        ("RSCode", "rs_code", "RSCode"),
        ("BinarySymmetricChannel", "binary_symmetric_channel", "BSC"),
    ],
)
@pytest.mark.parametrize("modern_first", [False, True])
def test_legacy_and_package_imports_share_classes_and_state(legacy, modern, name, modern_first):
    package_name = "coding_theory_algorithms." + modern
    first, second = (package_name, legacy) if modern_first else (legacy, package_name)
    statement = f"""
from importlib import import_module
first = import_module({first!r})
second = import_module({second!r})
assert getattr(first, {name!r}) is getattr(second, {name!r})
first.compatibility_probe = object()
assert second.compatibility_probe is first.compatibility_probe
del first.compatibility_probe
"""
    result = run_python("-c", statement)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_binary_field_singleton_is_shared_across_import_paths():
    result = run_python(
        "-c",
        "from GaloisField import GF2; "
        "from coding_theory_algorithms.galois_field import GF2 as modern; "
        "from coding_theory_algorithms.cyclic_code import GF2 as cyclic; "
        "assert GF2 is modern is cyclic; "
        "assert GF2.multElements(1, 1) == 1",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("args", [("BinarySymmetricChannel.py",), ("-m", "coding_theory_algorithms.binary_symmetric_channel")])
def test_channel_script_entry_points(args):
    result = run_python(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    capacity_lines = [line for line in result.stdout.splitlines() if line.startswith("Cs = Imax(X,Y) = ")]
    assert len(capacity_lines) == 1, result.stdout
    capacity = float(capacity_lines[0].split(" = ")[-1].removesuffix(" bits"))
    assert capacity == pytest.approx(2 / 3)
    assert result.stderr == ""
