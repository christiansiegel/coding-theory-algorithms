"""Protect the documented script command and existing module imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_python(*args):
    # Exercise repository-local imports without an inherited PYTHONPATH.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "statement",
    [
        "from BinarySymmetricChannel import BSC",
        "from GaloisField import GaloisField, GF2, X, degree",
        "from LinearBlockCode import LinearBlockCode",
        "from CyclicCode import CyclicCode",
        "from BCHCode import BCHCode",
        "from RSCode import RSCode",
    ],
)
def test_existing_imports(statement):
    result = run_python("-c", statement)
    assert result.returncode == 0, result.stdout + result.stderr


def test_readme_script_runs_both_examples():
    result = run_python("playground.py")
    assert result.returncode == 0, result.stdout + result.stderr
    # Check hand-verifiable results while allowing whitespace changes.
    assert "Exercise 1" in result.stdout
    assert "Exercise 2" in result.stdout
    lines = {" ".join(line.split()) for line in result.stdout.splitlines()}
    expected_results = {
        # alpha^5 = 1 + alpha^2 follows from the defining polynomial.
        "α^5 [1 0 1 0 0] 1 + α^2",
        # The generator has degree 4, so the (7, 3) code has rate 3/7.
        "-> Message length (k): 3",
        "-> Codeword length (n): 7",
        "-> Coding rate (R = k/n): 0.42857142857142855",
        # Its seven nonzero codewords each have weight 4: dmin=4 and t=1.
        "-> Minimum Distance (dmin): 4",
        "-> Error Correction Capability (t): 1",
        # Systematic encoding of 111: X^4*m + remainder(X^4*m / g).
        "[1 1 1] [0 0 1 0 1 1 1] m(X) = 1 + X + X^2 c(X) = X^2 + X^4 + X^5 + X^6",
    }
    assert expected_results <= lines, f"Missing results: {expected_results - lines}"

    # The first information bit is position 4 (G = [P | I]); its syndrome
    # is column 4 of H. Scope the assertion to the decoding table.
    decoding_table = result.stdout.split("Correctable Error Patterns -> Syndromes")
    assert len(decoding_table) == 2, result.stdout
    decoding_lines = {" ".join(line.split()) for line in decoding_table[1].splitlines()}
    assert "[0 0 0 0 1 0 0] [1 0 1 1]" in decoding_lines
