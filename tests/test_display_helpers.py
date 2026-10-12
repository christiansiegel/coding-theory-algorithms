"""Check matrix row spaces and diagnostic mathematics independently."""

from itertools import product
import re

import numpy as np
import pytest

from coding_theory_algorithms.cyclic_code import makeSystematic, printAllCyclicCodes
from coding_theory_algorithms.linear_block_code import matrixMultiplicationEquations

from .test_cyclic_codes import carryless_product, vector


def row_space(matrix):
    return {tuple(sum(bit * row[column] for bit, row in zip(message, matrix)) % 2 for column in range(matrix.shape[1])) for message in product([0, 1], repeat=matrix.shape[0])}


@pytest.mark.parametrize("matrix", [[[1, 0, 0, 1], [0, 1, 1, 0]], [[1, 0, 1, 1], [0, 1, 0, 1]], [[1, 0, 1, 1, 0], [0, 1, 0, 1, 1], [1, 1, 1, 0, 0]]])
def test_systematic_conversion_preserves_row_space_and_input(matrix, capsys):
    original = np.array(matrix)
    for verbose in [False, True]:
        result = makeSystematic(original, verbose)
        np.testing.assert_array_equal(result[:, -len(matrix) :], np.eye(len(matrix), dtype=int))
        assert row_space(result) == row_space(original)
        np.testing.assert_array_equal(original, matrix)
        assert bool(capsys.readouterr().out) == verbose


def test_systematic_conversion_rejects_singular_information_block():
    original = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    with pytest.raises(ValueError, match="invertible"):
        makeSystematic(original, False)
    np.testing.assert_array_equal(original, [[1, 0, 0, 0], [0, 1, 0, 0]])


def test_printed_matrix_equations_match_every_input():
    matrix = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
    equations = matrixMultiplicationEquations(matrix, "c", "m").strip().splitlines()
    assert len(equations) == 4
    for message in product([0, 1], repeat=2):
        for column, equation in enumerate(equations):
            left, right = equation.split(" = ")
            assert left == f"c{column}"
            terms = right.split(" ⊕ ")
            actual = 0 if terms == ["0"] else sum(message[int(term[1:])] for term in terms) % 2
            assert actual == sum(message[row] * matrix[row, column] for row in range(2)) % 2


def test_printed_cyclic_code_dimensions_match_factor_products(capsys):
    factors = [0b11, 0b1011, 0b1101]
    printAllCyclicCodes([vector(factor, factor.bit_length()) for factor in factors])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Ccyc(")]
    assert len(lines) == 6
    for subset, line in enumerate(lines, start=1):
        generator = 1
        for index, factor in enumerate(factors):
            if subset & (1 << index):
                generator = carryless_product(generator, factor)
        dimensions = re.match(r"Ccyc\((\d+), (\d+)\)", line)
        assert tuple(map(int, dimensions.groups())) == (7, 7 - (generator.bit_length() - 1))
