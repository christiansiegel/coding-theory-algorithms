"""Verify element arithmetic against independent binary-polynomial arithmetic."""

import numpy as np
import pytest

from GaloisField import GaloisField


def multiply_bits(a, b, modulus):
    """Carryless multiplication reduced by a binary polynomial bit mask."""
    product = 0
    high_bit = 1 << (modulus.bit_length() - 1)
    while b:
        if b & 1:
            product ^= a
        b >>= 1
        a <<= 1
        if a & high_bit:
            a ^= modulus
    return product


@pytest.fixture(
    params=[0b11, 0b111, 0b1011, 0b10011], ids=["GF2", "GF4", "GF8", "GF16"]
)
def field_case(request):
    modulus = request.param
    m = modulus.bit_length() - 1
    coefficients = np.array([(modulus >> i) & 1 for i in range(m + 1)])
    field = GaloisField(coefficients)
    # Independent label -> binary polynomial table: 0, 1, alpha, alpha^2, ...
    bits = [0]
    value = 1
    for _ in range((1 << m) - 1):
        bits.append(value)
        value = multiply_bits(value, 2, modulus)
    assert value == 1
    assert len(set(bits)) == 1 << m
    labels = {value: label for label, value in enumerate(bits)}
    return field, modulus, bits, labels


def test_reference_gf16_known_product():
    # alpha^4 = 1 + alpha under 1 + X + X^4.
    assert multiply_bits(0b1000, 0b0010, 0b10011) == 0b0011


def test_dimensions_and_defining_polynomial(field_case):
    field, modulus, bits, _ = field_case
    m = modulus.bit_length() - 1
    assert field.m() == m
    assert field.q() == len(bits) == 1 << m
    np.testing.assert_array_equal(field.p(), [(modulus >> i) & 1 for i in range(m + 1)])


def test_element_conversions(field_case):
    field, _, bits, _ = field_case
    for label, value in enumerate(bits):
        vector = np.array([(value >> i) & 1 for i in range(field.m())])
        np.testing.assert_array_equal(field.elementToPoly(label), vector)
        np.testing.assert_array_equal(field.elementToPoly(label, True), vector)
        assert field.elementFromPoly(vector) == label
        assert field.elementFromPoly(np.pad(vector, (0, 2))) == label
        assert field.elementToExp(label) == (label - 1 if label else -float("inf"))


def test_label_normalization(field_case):
    field, _, bits, _ = field_case
    period = len(bits) - 1
    assert field.element(0) == 0
    for label in range(-2 * period, 3 * period + 1):
        if label:
            assert field.element(label) == 1 + (label - 1) % period


def test_nonnegative_exponents_and_infinity(field_case):
    field, _, bits, _ = field_case
    period = len(bits) - 1
    for exponent in range(3 * period):
        assert field.elementFromExp(exponent) == 1 + exponent % period
    assert field.elementFromExp(float("inf")) == 0
    assert field.elementFromExp(-float("inf")) == 0


def test_addition_and_multiplication_exhaustively(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits):
        for b, value_b in enumerate(bits):
            assert field.addElements(a, b) == labels[value_a ^ value_b], (a, b)
            assert (
                field.multElements(a, b)
                == labels[multiply_bits(value_a, value_b, modulus)]
            ), (a, b)


def test_division_by_nonzero_elements_exhaustively(field_case):
    field, modulus, bits, labels = field_case
    # Find inverses by independent multiplication rather than production division.
    for b, value_b in enumerate(bits[1:], start=1):
        inverse = next(v for v in bits[1:] if multiply_bits(value_b, v, modulus) == 1)
        for a, value_a in enumerate(bits):
            if a and a - b == -1:
                # Covered separately: exponent -1 is a confirmed conversion bug.
                continue
            expected = labels[multiply_bits(value_a, inverse, modulus)]
            assert field.divElements(a, b) == expected, (a, b)


def test_positive_powers_and_nonzero_power_zero(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits):
        value = 1
        if a:
            assert field.powElement(a, 0) == 1
        for exponent in range(1, 2 * len(bits)):
            value = multiply_bits(value, value_a, modulus)
            assert field.powElement(a, exponent) == labels[value], (a, exponent)


def test_nonzero_negative_powers(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits[1:], start=1):
        inverse = next(v for v in bits[1:] if multiply_bits(value_a, v, modulus) == 1)
        value = 1
        for exponent in range(1, len(bits)):
            value = multiply_bits(value, inverse, modulus)
            if a == 2 and exponent == 1:
                # Covered separately: alpha^-1 hits the conversion bug.
                continue
            assert field.powElement(a, -exponent) == labels[value], (a, -exponent)


def test_division_requiring_exponent_minus_one(field_case, request):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.divElements(1, 1) == 1
        return
    request.applymarker(
        pytest.mark.xfail(strict=True, reason="division with exponent -1 returns zero")
    )
    assert field.divElements(1, 2) == len(bits) - 1


def test_alpha_inverse_power(field_case, request):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.powElement(1, -1) == 1
        return
    request.applymarker(
        pytest.mark.xfail(strict=True, reason="alpha^-1 incorrectly returns zero")
    )
    assert field.powElement(2, -1) == len(bits) - 1


@pytest.mark.xfail(
    strict=True, reason="elementFromExp(-1) incorrectly maps alpha's inverse to zero"
)
def test_negative_exponent_conversion(field_case):
    field, _, bits, _ = field_case
    assert field.elementFromExp(-1) == len(bits) - 1


@pytest.mark.xfail(strict=True, reason="divElements does not reject a zero denominator")
def test_division_by_zero_raises(field_case):
    field, _, bits, _ = field_case
    for a in range(len(bits)):
        with pytest.raises(ZeroDivisionError):
            field.divElements(a, 0)


@pytest.mark.xfail(strict=True, reason="powElement(0, 0) computes infinity times zero")
def test_zero_power_zero_is_one(field_case):
    field, _, _, _ = field_case
    assert field.powElement(0, 0) == 1


@pytest.mark.xfail(
    strict=True, reason="powElement does not reject negative powers of zero"
)
def test_zero_negative_power_raises(field_case):
    field, _, _, _ = field_case
    with pytest.raises(ZeroDivisionError):
        field.powElement(0, -1)
