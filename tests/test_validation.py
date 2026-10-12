"""Public boundaries reject malformed inputs before arithmetic or allocation."""

import numpy as np
import pytest

from BinarySymmetricChannel import BSC
from GaloisField import GaloisField
from LinearBlockCode import LinearBlockCode
from CyclicCode import CyclicCode
from BCHCode import BCHCode
from RSCode import RSCode


@pytest.mark.parametrize(
    "source,transition",
    [([], []), ([1], []), ([0.5, 0.5], [[1, 0]]), ([[1]], [[1]]), ([1], [[0.2, 0.2]]), ([0.2, 0.2], [[1], [1]]), ([-1, 2], [[1], [1]]), ([1], [[-1, 2]]), ([np.nan], [[1]]), ([1], [[np.inf]])],
)
def test_invalid_channels(source, transition):
    with pytest.raises(ValueError):
        BSC(source, transition)


@pytest.mark.parametrize("polynomial", [[], [0], [1], [0, 1], [1, 0, 1], [1, 1, 1, 1, 1], [1, 0.5, 1], [[1, 1]], [1, np.nan]])
def test_invalid_field_polynomials(polynomial):
    with pytest.raises(ValueError):
        GaloisField(polynomial)


def test_field_constructor_copies_and_trims_input():
    polynomial = np.array([1, 1, 0, 1, 0])
    field = GaloisField(polynomial)
    polynomial[:] = 0
    np.testing.assert_array_equal(field.p(), [1, 1, 0, 1])
    assert field.q() == 8


@pytest.mark.parametrize("generator,length", [([], 7), ([1], 7), ([0, 1], 7), ([1, 2, 1], 7), ([1, 1], 0), ([1, 1], 2.5), ([1, 1, 1], 7), ([1, 0, 1], 2)])
def test_invalid_cyclic_parameters(generator, length):
    with pytest.raises(ValueError):
        CyclicCode(generator, length)


@pytest.mark.parametrize("code_type", [BCHCode, RSCode])
@pytest.mark.parametrize("radius", [-1, 0, 1.5, True, 4])
def test_invalid_design_radius(code_type, radius):
    field = GaloisField([1, 1, 0, 1])
    with pytest.raises(ValueError):
        code_type(field, radius)


@pytest.mark.parametrize("matrix", [[], [[1]], [[0, 0, 0]], [[1, 1, 0]], [[1, 2, 1]], [[1, 0.5, 1]]])
def test_invalid_generator_does_not_change_existing_code(matrix):
    code = LinearBlockCode()
    code.setG([[1, 1, 1]])
    with pytest.raises(ValueError):
        code.setG(matrix)
    np.testing.assert_array_equal(code.G(), [[1, 1, 1]])


@pytest.mark.parametrize("matrix", [[], [[1]], [[0, 1, 1]], [[1, 2, 1]], [[0.5, 1, 1]]])
def test_invalid_parity_checks(matrix):
    with pytest.raises(ValueError):
        LinearBlockCode().setH(matrix)


@pytest.mark.parametrize("p", [-0.01, 1.01, np.nan, np.inf, "0.5"])
@pytest.mark.parametrize("method", ["PU", "Pe"])
def test_invalid_error_probabilities(method, p):
    code = LinearBlockCode()
    code.setG([[1, 1, 1]])
    with pytest.raises(ValueError):
        getattr(code, method)(p)


@pytest.mark.parametrize("message", [[], [[0]], [0.5], [2], [0, 0]])
def test_invalid_binary_messages(message):
    code = LinearBlockCode()
    code.setG([[1, 1, 1]])
    with pytest.raises(ValueError):
        code.c(message)


@pytest.mark.parametrize("method", ["c", "S", "decode"])
@pytest.mark.parametrize("argument", [[], [[0]], [0.5], [4], [0, 0, 0, 0]])
def test_invalid_rs_vectors(method, argument):
    code = RSCode(GaloisField([1, 1, 1]), 1)
    with pytest.raises(ValueError):
        getattr(code, method)(argument)


def test_valid_lists_and_constructor_copies():
    source, transition = [0.25, 0.75], [[1, 0], [0, 1]]
    channel = BSC(source, transition)
    source[0] = 1
    transition[0][0] = 0
    np.testing.assert_array_equal(channel.PX(), [0.25, 0.75])
    np.testing.assert_array_equal(channel.PYX(), np.eye(2))
    code = CyclicCode([1, 1], 3)
    np.testing.assert_array_equal(code.c([1]), [1, 1, 0])
    with pytest.raises(NotImplementedError):
        code.setG([[1, 1, 1]])
