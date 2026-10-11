"""Verify element arithmetic against independent binary-polynomial arithmetic."""

import json
from pathlib import Path
import subprocess
import sys

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
            assert field.powElement(a, -exponent) == labels[value], (a, -exponent)


def test_division_requiring_exponent_minus_one(field_case):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.divElements(1, 1) == 1
        return
    assert field.divElements(1, 2) == len(bits) - 1


def test_alpha_inverse_power(field_case):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.powElement(1, -1) == 1
        return
    assert field.powElement(2, -1) == len(bits) - 1


def test_negative_exponent_conversion(field_case):
    field, _, bits, _ = field_case
    period = len(bits) - 1
    for exponent in range(-3 * period, 0):
        assert field.elementFromExp(exponent) == 1 + exponent % period


def test_division_by_zero_raises(field_case):
    field, _, bits, _ = field_case
    for a in range(len(bits)):
        with pytest.raises(ZeroDivisionError):
            field.divElements(a, 0)


def test_zero_power_zero_is_one(field_case):
    field, _, _, _ = field_case
    assert field.powElement(0, 0) == 1


def test_zero_negative_power_raises(field_case):
    field, _, _, _ = field_case
    with pytest.raises(ZeroDivisionError):
        field.powElement(0, -1)


def trim_polynomial(coefficients):
    result = list(coefficients)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def reference_poly_product(a, b, modulus):
    result = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] ^= multiply_bits(x, y, modulus)
    return trim_polynomial(result)


def reference_poly_division(a, b, modulus):
    residual = trim_polynomial(a)
    divisor = trim_polynomial(b)
    assert divisor != [0]
    inverse = next(
        value
        for value in range(1, 1 << (modulus.bit_length() - 1))
        if multiply_bits(divisor[-1], value, modulus) == 1
    )
    quotient = [0] * max(1, len(residual) - len(divisor) + 1)
    while residual != [0] and len(residual) >= len(divisor):
        offset = len(residual) - len(divisor)
        factor = multiply_bits(residual[-1], inverse, modulus)
        quotient[offset] ^= factor
        for i, value in enumerate(divisor):
            residual[offset + i] ^= multiply_bits(factor, value, modulus)
        residual = trim_polynomial(residual)
    return trim_polynomial(quotient), residual


def as_polynomial_bits(actual, bits):
    if actual.size:
        assert np.issubdtype(actual.dtype, np.integer)
    return trim_polynomial([bits[int(label)] for label in actual])


def test_polynomial_addition_and_multiplication(field_case):
    field, modulus, bits, labels = field_case
    samples = [[0], [1], [0, 1], [1, 1, 0], [bits[-1], 1], [1, bits[-1], 1]]
    for a in samples:
        for b in samples:
            operand_a = np.array([labels[v] for v in a])
            operand_b = np.array([labels[v] for v in b])
            before_a, before_b = operand_a.copy(), operand_b.copy()
            expected_sum = [
                (a[i] if i < len(a) else 0) ^ (b[i] if i < len(b) else 0)
                for i in range(max(len(a), len(b)))
            ]
            assert as_polynomial_bits(
                field.addPoly(operand_a, operand_b), bits
            ) == trim_polynomial(expected_sum)
            assert as_polynomial_bits(
                field.multPoly(operand_a, operand_b), bits
            ) == reference_poly_product(a, b, modulus)
            np.testing.assert_array_equal(operand_a, before_a)
            np.testing.assert_array_equal(operand_b, before_b)


def test_polynomial_division_by_nonconstant_divisors(field_case):
    field, modulus, bits, labels = field_case
    dividends = [[0], [1], [1, 0, 1], [bits[-1], 1, 1], [1, 1, 0, 1]]
    divisors = [[1, 1], [bits[-1], bits[-1]], [1, 0, bits[-1]]]
    for a in dividends:
        for b in divisors:
            operand_a = np.array([labels[v] for v in a])
            operand_b = np.array([labels[v] for v in b])
            before_a, before_b = operand_a.copy(), operand_b.copy()
            expected_q, expected_r = reference_poly_division(a, b, modulus)
            q, r = field.divmodPoly(operand_a, operand_b)
            assert as_polynomial_bits(q, bits) == expected_q
            assert as_polynomial_bits(r, bits) == expected_r
            assert (
                as_polynomial_bits(field.divPoly(operand_a, operand_b), bits)
                == expected_q
            )
            assert (
                as_polynomial_bits(field.modPoly(operand_a, operand_b), bits)
                == expected_r
            )
            assert field.isFactor(operand_a, operand_b) == (expected_r == [0])
            np.testing.assert_array_equal(operand_a, before_a)
            np.testing.assert_array_equal(operand_b, before_b)


def test_polynomial_evaluation(field_case):
    field, modulus, bits, labels = field_case
    for coefficients in [[0], [bits[-1]], [1, bits[-1], 1], [0, 1, 1, 1]]:
        p = np.array([labels[v] for v in coefficients])
        before = p.copy()
        for a, value in enumerate(bits):
            expected = 0
            for coefficient in reversed(coefficients):
                expected = multiply_bits(expected, value, modulus) ^ coefficient
            assert field.substituteElementIntoPoly(p, a) == labels[expected]
        np.testing.assert_array_equal(p, before)


def test_monic_multiplier(field_case):
    field, modulus, bits, labels = field_case
    coefficient = bits[-1]
    inverse = next(
        value for value in bits[1:] if multiply_bits(coefficient, value, modulus) == 1
    )
    assert field.monicMultiplier(np.array([1, labels[coefficient]])) == labels[inverse]


@pytest.mark.xfail(
    strict=True,
    reason="monicMultiplier uses the final array entry instead of the leading nonzero coefficient",
)
def test_monic_multiplier_with_trailing_zeros(field_case):
    field, modulus, bits, labels = field_case
    coefficient = bits[-1]
    inverse = next(
        value for value in bits[1:] if multiply_bits(coefficient, value, modulus) == 1
    )
    assert (
        field.monicMultiplier(np.array([1, labels[coefficient], 0])) == labels[inverse]
    )


@pytest.mark.xfail(
    strict=True, reason="a zero quotient is returned as an empty floating-point array"
)
def test_zero_quotient_has_integer_dtype(field_case):
    field, _, _, _ = field_case
    q, _ = field.divmodPoly(np.array([1]), np.array([1, 1]))
    assert np.issubdtype(q.dtype, np.integer)


@pytest.mark.parametrize("coefficients", [[0], [1], [0, 1], [1, 1, 1, 1]])
@pytest.mark.xfail(
    strict=True,
    reason="derivePoly omits the highest term and mishandles characteristic two",
)
def test_formal_polynomial_derivative(field_case, coefficients):
    field, _, _, _ = field_case
    p = np.array(coefficients)
    expected = [p[i] if i % 2 else 0 for i in range(1, len(p))]
    np.testing.assert_array_equal(
        trim_polynomial(field.derivePoly(p)), trim_polynomial(expected)
    )
    assert np.issubdtype(field.derivePoly(p).dtype, np.integer)


@pytest.mark.parametrize(
    "method, expected", [("addPoly", [0, 1]), ("multPoly", [1, 1])]
)
@pytest.mark.xfail(
    strict=True,
    reason="scalar second operand is mistakenly assigned to the first operand",
)
def test_scalar_second_polynomial_operand(field_case, method, expected):
    field, _, _, _ = field_case
    np.testing.assert_array_equal(getattr(field, method)(np.array([1, 1]), 1), expected)


@pytest.mark.xfail(
    strict=True,
    reason="polynomial zero divisors raise AssertionError instead of ZeroDivisionError",
)
def test_polynomial_division_by_zero_raises(field_case):
    field, _, _, _ = field_case
    with pytest.raises(ZeroDivisionError):
        field.divmodPoly(np.array([1, 1]), np.array([0]))


@pytest.mark.parametrize("dividend", [[0], [1, 0, 1]])
@pytest.mark.xfail(
    strict=True,
    raises=subprocess.TimeoutExpired,
    reason="division by a constant polynomial does not terminate",
)
def test_constant_polynomial_division_terminates(dividend):
    # Run in a child process so a nonterminating algorithm cannot hang pytest.
    script = (
        "import json; import numpy as np; from GaloisField import GF2; "
        f"q, r = GF2.divmodPoly(np.array({dividend!r}), np.array([1])); "
        "print(json.dumps([q.tolist(), r.tolist()]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    q, r = json.loads(result.stdout)
    assert trim_polynomial(q) == dividend
    assert trim_polynomial(r) == [0]
