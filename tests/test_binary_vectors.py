"""Known answers and independent checks for integer-to-bit conversion."""

import warnings

import numpy as np
import pytest

from coding_theory_algorithms.linear_block_code import int_to_array


@pytest.mark.parametrize(
    "value, length, expected",
    [
        (0, 0, [0]),
        (1, 0, [1]),
        (2, 0, [0, 1]),
        (13, 0, [1, 0, 1, 1]),
        (0, 4, [0, 0, 0, 0]),
        (13, 6, [1, 0, 1, 1, 0, 0]),
    ],
)
def test_int_to_array_known_answers(value, length, expected):
    # Removed NumPy APIs should fail this test even on versions that only warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = int_to_array(value, length)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def test_int_to_array_all_eight_bit_values():
    for value in range(256):
        expected = [(value >> position) & 1 for position in range(8)]
        np.testing.assert_array_equal(int_to_array(value, 8), expected)


def test_int_to_array_large_integer():
    value = (1 << 80) | (1 << 40) | 1
    expected = [(value >> position) & 1 for position in range(81)]
    np.testing.assert_array_equal(int_to_array(value), expected)
