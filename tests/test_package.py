"""Package entry points and shared binary-field state."""

import pytest

from .test_smoke import run_python


def test_package_exports_match_submodules_and_support_star_import():
    result = run_python(
        "-c",
        """
import importlib
import coding_theory_algorithms as package
modules = {
    'BSC': 'binary_symmetric_channel', 'GaloisField': 'galois_field',
    'GF2': 'galois_field', 'LinearBlockCode': 'linear_block_code',
    'CyclicCode': 'cyclic_code', 'BCHCode': 'bch_code', 'RSCode': 'rs_code',
}
assert set(package.__all__) == set(modules)
assert set(modules) <= set(dir(package))
namespace = {}
exec('from coding_theory_algorithms import *', namespace)
assert set(namespace) - {'__builtins__'} == set(modules)
for name, module in modules.items():
    implementation = importlib.import_module('coding_theory_algorithms.' + module)
    assert namespace[name] is getattr(implementation, name)
    assert getattr(package, name) is namespace[name]
assert package.GF2.mult_elements(1, 1) == 1
assert not hasattr(package, 'unknown_export')
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == result.stderr == ""


def test_binary_field_singleton_is_shared_by_internal_imports():
    result = run_python(
        "-c", "from coding_theory_algorithms.galois_field import GF2; from coding_theory_algorithms.cyclic_code import GF2 as cyclic; assert GF2 is cyclic; assert GF2.mult_elements(1, 1) == 1"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_channel_module_entry_point():
    result = run_python("-m", "coding_theory_algorithms.binary_symmetric_channel")
    assert result.returncode == 0, result.stdout + result.stderr
    capacity_lines = [line for line in result.stdout.splitlines() if line.startswith("Cs = Imax(X,Y) = ")]
    assert len(capacity_lines) == 1, result.stdout
    capacity = float(capacity_lines[0].split(" = ")[-1].removesuffix(" bits"))
    assert capacity == pytest.approx(2 / 3)
    assert result.stderr == ""
