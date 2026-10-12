"""Check cyclic codes using binary polynomial arithmetic independent of NumPy."""

import numpy as np
import pytest

from coding_theory_algorithms.cyclic_code import CyclicCode


def vector(value, length):
    return np.array([(value >> i) & 1 for i in range(length)])


def carryless_product(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def remainder(a, b):
    while a and a.bit_length() >= b.bit_length():
        a ^= b << (a.bit_length() - b.bit_length())
    return a


@pytest.fixture(params=[(7, 0b11101), (7, 0b1011), (5, 0b11111), (7, 0b11)])
def cyclic_case(request):
    n, g = request.param
    assert remainder((1 << n) | 1, g) == 0
    code = CyclicCode(vector(g, g.bit_length()), n)
    return code, n, g


def test_encoding_in_both_forms_and_generator_matrices(cyclic_case):
    code, n, g = cyclic_case
    k = n - (g.bit_length() - 1)
    assert code.k() == k
    assert code.n() == n
    systematic_matrix = code.G()
    np.testing.assert_array_equal(systematic_matrix[:, -k:], np.eye(k, dtype=int))
    np.testing.assert_array_equal(systematic_matrix @ code.H().T % 2, 0)
    for message in range(1 << k):
        m = vector(message, k)
        shifted = message << (n - k)
        expected_systematic = vector(shifted ^ remainder(shifted, g), n)
        expected_plain = vector(carryless_product(message, g), n)
        np.testing.assert_array_equal(code.c(m), expected_systematic)
        np.testing.assert_array_equal(code.c(m, False), expected_plain)
        np.testing.assert_array_equal(m @ systematic_matrix % 2, expected_systematic)
        np.testing.assert_array_equal(m @ code.G(False) % 2, expected_plain)
        np.testing.assert_array_equal(code.s(expected_systematic), np.zeros(n - k))


def test_polynomial_syndromes_for_every_received_vector(cyclic_case):
    code, n, g = cyclic_case
    for received in range(1 << n):
        actual = code.S(vector(received, n))
        # Polynomial syndrome length may vary: compare values, not zero padding.
        actual_value = sum(int(bit) << i for i, bit in enumerate(actual))
        assert actual_value == remainder(received, g)


def test_distance_in_small_cyclic_examples(cyclic_case):
    code, _, g = cyclic_case
    expected = min(carryless_product(message, g).bit_count() for message in range(1, 1 << code.k()))
    assert code.dmin() == expected
    assert code.t() == (expected - 1) // 2
    assert code.error_detection_capability() == expected - 1


def test_cyclic_shifts_for_all_messages(cyclic_case):
    code, n, g = cyclic_case
    for message in range(1 << code.k()):
        c = code.c(vector(message, code.k()))
        value = sum(int(bit) << i for i, bit in enumerate(c))
        for amount in [0, 1, n - 1, n, n + 1]:
            expected = vector(remainder(value << amount, (1 << n) | 1), n)
            np.testing.assert_array_equal(code.shift(c, amount), expected)


@pytest.mark.parametrize("amount_kind", ["one", "n_minus_one", "n", "n_plus_one"])
def test_cyclic_shift_with_wraparound(cyclic_case, amount_kind):
    code, n, _ = cyclic_case
    message = np.zeros(code.k(), dtype=int)
    message[-1] = 1
    codeword = code.c(message)
    value = sum(int(bit) << i for i, bit in enumerate(codeword))
    amount = {"one": 1, "n_minus_one": n - 1, "n": n, "n_plus_one": n + 1}[amount_kind]
    expected = vector(remainder(value << amount, (1 << n) | 1), n)
    np.testing.assert_array_equal(code.shift(codeword, amount), expected)


def test_shifted_polynomial_syndromes(cyclic_case):
    code, n, g = cyclic_case
    received = 0b101
    for amount in [0, 1, n, n + 1]:
        actual = code.shift_syndrome(code.S(vector(received, n)), amount)
        actual_value = sum(int(bit) << i for i, bit in enumerate(actual))
        assert actual_value == remainder(received << amount, g)


@pytest.mark.parametrize("method, expected", [("dmin", 2), ("t", 0), ("error_detection_capability", 1)])
def test_generator_weight_is_not_the_minimum_distance(method, expected):
    # g=1+X+X^2+X^3+X^4 divides X^15+1. Its weight is 5, but
    # (1+X)*g = 1+X^5 is a weight-2 codeword. No monomial is divisible by g.
    code = CyclicCode(np.ones(5, dtype=int), 15)
    np.testing.assert_array_equal(code.c(vector(0b11, 11), False), vector(0b100001, 15))
    assert getattr(code, method)() == expected


def test_distance_two_code_does_not_attempt_single_error_correction():
    code = CyclicCode(np.ones(5, dtype=int), 15)
    received = vector(1, 15)
    with pytest.raises(KeyError):
        code.syndrome_decode(received)
