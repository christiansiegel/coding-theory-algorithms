"""Independent GCD answers and extended-Euclid decoder identities."""

import math

import numpy as np
import pytest

from BCHCode import HCF
from GaloisField import GaloisField

from .field_reference import multiply_bits, reference_poly_division, reference_poly_product, trim_polynomial


def test_integer_gcd_for_small_nonnegative_operands(capsys):
    for a in range(26):
        for b in range(26):
            if a == b == 0:
                continue
            assert HCF(a, b) == math.gcd(a, b)
    assert capsys.readouterr().out


@pytest.mark.parametrize("a,b", [(0, 0), (-15, 6), (15, -6), (-15, -6), (2**54 + 1, 3), (10**400, 3)])
@pytest.mark.xfail(strict=True, reason="integer Euclid mishandles zero/signs and uses floating-point division")
def test_integer_gcd_edge_cases(a, b):
    assert HCF(a, b) == math.gcd(a, b)


@pytest.mark.parametrize(
    "modulus,a,b,expected_r,expected_t",
    [
        (0b11, [1, 0, 0, 0, 1], [1, 0, 1], [0], [1, 0, 1]),
        (0b111, [0, 0, 1], [2, 1], [3], [2, 1]),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_polynomial_euclid_known_answers(modulus, a, b, expected_r, expected_t, reverse, capsys):
    field = GaloisField(np.array([(modulus >> i) & 1 for i in range(modulus.bit_length())]))
    first, second = (b, a) if reverse else (a, b)
    r, t = field.HCF(np.array(first), np.array(second))
    assert trim_polynomial(r) == expected_r
    assert trim_polynomial(t) == expected_t
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("modulus", [0b11, 0b111, 0b1011])
def test_polynomial_euclid_modular_identity_and_input_preservation(modulus, capsys):
    field = GaloisField(np.array([(modulus >> i) & 1 for i in range(modulus.bit_length())]))
    # Build label-to-bit values from the defining polynomial independently.
    bits = [0]
    value = 1
    for _ in range(field.q() - 1):
        bits.append(value)
        value = multiply_bits(value, 2, modulus)
    a = np.array([0, 0, 0, 0, 1, 0, 0])
    messages = [[1], [0, 1], [1, 1], [1, 0, 1], [field.q() - 1, 1, field.q() - 1]]
    for coefficients in messages:
        b = np.array(coefficients + [0, 0])
        before_a, before_b = a.copy(), b.copy()
        r, t = field.HCF(a, b)
        r_bits = trim_polynomial([bits[int(v)] for v in r])
        t_bits = trim_polynomial([bits[int(v)] for v in t])
        product = reference_poly_product(t_bits, [bits[int(v)] for v in b], modulus)
        _, residual = reference_poly_division(product, [0, 0, 0, 0, 1], modulus)
        assert r_bits == residual
        assert t_bits != [0]
        assert r_bits == [0] or len(r_bits) < len(t_bits)
        np.testing.assert_array_equal(a, before_a)
        np.testing.assert_array_equal(b, before_b)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("a,b,expected_t", [([1, 1], [1, 1], [1]), ([1], [1], [1])])
@pytest.mark.xfail(strict=True, reason="zero remainder does not stop when t is constant")
def test_polynomial_euclid_zero_remainder_with_constant_coefficient(a, b, expected_t):
    field = GaloisField()
    r, t = field.HCF(np.array(a), np.array(b))
    assert trim_polynomial(r) == [0]
    assert trim_polynomial(t) == expected_t


@pytest.mark.parametrize("a,b", [([0], [1, 1]), ([1, 1], [0]), ([0], [0])])
@pytest.mark.xfail(strict=True, reason="zero operands reach division by zero instead of input validation")
def test_polynomial_euclid_rejects_zero_operands(a, b):
    with pytest.raises(ValueError):
        GaloisField().HCF(np.array(a), np.array(b))
