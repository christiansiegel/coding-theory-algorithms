"""Verify element arithmetic against independent binary-polynomial arithmetic."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from coding_theory_algorithms.galois_field import GaloisField

from .field_reference import (
    multiply_bits,
    reference_poly_division,
    reference_poly_product,
    trim_polynomial,
)


@pytest.fixture(params=[0b11, 0b111, 0b1011, 0b10011], ids=["GF2", "GF4", "GF8", "GF16"])
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


def reference_conjugate_group(root, modulus, bits, labels):
    group = set()
    value = bits[root]
    while labels[value] not in group:
        group.add(labels[value])
        value = multiply_bits(value, value, modulus)
    return sorted(group)


def reference_evaluate(coefficients, value, modulus):
    result = 0
    for coefficient in reversed(coefficients):
        result = multiply_bits(result, value, modulus) ^ coefficient
    return result


def reference_binary_remainder(polynomial, divisor):
    while polynomial and polynomial.bit_length() >= divisor.bit_length():
        polynomial ^= divisor << (polynomial.bit_length() - divisor.bit_length())
    return polynomial


def reference_binary_classification(polynomial):
    degree = polynomial.bit_length() - 1
    if degree < 1:
        return False, False
    # A reducible polynomial has a monic factor of degree at most half its own.
    for factor_degree in range(1, degree // 2 + 1):
        for factor in range(1 << factor_degree, 1 << (factor_degree + 1)):
            if reference_binary_remainder(polynomial, factor) == 0:
                return False, False
    # Compute the order of X in the quotient using integer polynomial values.
    value = 1
    for exponent in range(1, 1 << degree):
        value = multiply_bits(value, 2, polynomial)
        if value == 1:
            return True, exponent == (1 << degree) - 1
    return True, False


@pytest.mark.parametrize(
    "polynomial,expected",
    [(0, (False, False)), (1, (False, False)), (0b10, (True, False)), (0b11, (True, True)), (0b101, (False, False)), (0b111, (True, True)), (0b10011, (True, True)), (0b11111, (True, False))],
)
def test_binary_classification_reference_known_answers(polynomial, expected):
    assert reference_binary_classification(polynomial) == expected


@pytest.mark.parametrize("method,index", [("irreducible", 0), ("primitive", 1)])
def test_binary_polynomial_classification_independent_of_receiver(field_case, method, index, capsys):
    field, _, _, _ = field_case
    # Every binary polynomial of degree up to five, including constants/zero.
    for polynomial in range(64):
        coefficients = [(polynomial >> i) & 1 for i in range(max(1, polynomial.bit_length()))]
        expected = reference_binary_classification(polynomial)[index]
        for padding in [0, 2]:
            argument = np.array(coefficients + [0] * padding)
            before = argument.copy()
            for verbose in [False, True]:
                assert getattr(field, method)(argument, verbose) is expected, (method, field.m(), polynomial, padding)
                assert bool(capsys.readouterr().out) == verbose
                np.testing.assert_array_equal(argument, before)


def test_nonzero_roots_against_independent_evaluation(field_case):
    field, modulus, bits, _ = field_case
    # Include constants, X, repeated roots, and every monic linear/quadratic.
    polynomials = [[0], [1]] + [[a, 1] for a in range(len(bits))] + [[a, b, 1] for a in range(len(bits)) for b in range(len(bits))]
    for coefficients in polynomials:
        polynomial = np.array(coefficients)
        expected = [label for label in range(1, len(bits)) if reference_evaluate([bits[c] for c in coefficients], bits[label], modulus) == 0]
        assert field.roots(polynomial) == expected
        np.testing.assert_array_equal(polynomial, coefficients)
    assert field.roots(np.array([0, 1])) == []  # Zero is deliberately outside this search.


def test_conjugate_roots_against_independent_squaring(field_case, capsys):
    field, modulus, bits, labels = field_case
    for root in range(len(bits)):
        expected = reference_conjugate_group(root, modulus, bits, labels)
        for verbose in [False, True]:
            assert field.conjugate_roots(root, verbose) == expected
            assert bool(capsys.readouterr().out) == verbose


def test_conjugate_groups_partition_the_whole_field(field_case):
    field, modulus, bits, labels = field_case
    expected = sorted({tuple(reference_conjugate_group(root, modulus, bits, labels)) for root in range(len(bits))})
    actual = field.conjugate_root_groups()
    assert sorted(tuple(group) for group in actual) == expected
    flattened = [root for group in actual for root in group]
    assert sorted(flattened) == list(range(len(bits)))


def test_conjugate_reduction_preserves_inputs_and_supplied_representatives(field_case):
    field, modulus, bits, labels = field_case
    candidates = [[], list(range(len(bits))), list(reversed(range(len(bits)))), [1, 1, 0, 0]]
    candidates += [[root] for root in range(len(bits))]
    candidates += [[a, b, a] for a in range(len(bits)) for b in range(len(bits))]
    for roots in candidates:
        before = roots.copy()
        groups = {}
        for root in roots:
            key = tuple(reference_conjugate_group(root, modulus, bits, labels))
            groups[key] = min(root, groups.get(key, root))
        actual = field.remove_conjugate_roots(roots)
        assert roots == before
        assert actual == sorted(groups.values())
        assert actual is not roots


def test_minimal_polynomials_for_complete_conjugate_groups(field_case):
    field, modulus, bits, labels = field_case
    groups = {tuple(reference_conjugate_group(root, modulus, bits, labels)) for root in range(len(bits))}
    for group in groups:
        expected = [1]
        for root in group:
            expected = reference_poly_product(expected, [bits[root], 1], modulus)
        assert set(expected) <= {0, 1}
        for roots in [list(group), np.array(list(reversed(group)))]:
            before = roots.copy()
            actual = field.minimal_polynomial(roots)
            np.testing.assert_array_equal(actual, expected)
            assert np.issubdtype(actual.dtype, np.integer)
            assert len(actual) == len(group) + 1
            np.testing.assert_array_equal(roots, before)
            for root in group:
                assert reference_evaluate(actual, bits[root], modulus) == 0


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
        np.testing.assert_array_equal(field.element_to_poly(label), vector)
        np.testing.assert_array_equal(field.element_to_poly(label, True), vector)
        assert field.element_from_poly(vector) == label
        assert field.element_from_poly(np.pad(vector, (0, 2))) == label
        assert field.element_to_exp(label) == (label - 1 if label else -float("inf"))


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
        assert field.element_from_exp(exponent) == 1 + exponent % period
    assert field.element_from_exp(float("inf")) == 0
    assert field.element_from_exp(-float("inf")) == 0


def test_addition_and_multiplication_exhaustively(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits):
        for b, value_b in enumerate(bits):
            assert field.add_elements(a, b) == labels[value_a ^ value_b], (a, b)
            assert field.mult_elements(a, b) == labels[multiply_bits(value_a, value_b, modulus)], (a, b)


def test_division_by_nonzero_elements_exhaustively(field_case):
    field, modulus, bits, labels = field_case
    # Find inverses by independent multiplication rather than production division.
    for b, value_b in enumerate(bits[1:], start=1):
        inverse = next(v for v in bits[1:] if multiply_bits(value_b, v, modulus) == 1)
        for a, value_a in enumerate(bits):
            expected = labels[multiply_bits(value_a, inverse, modulus)]
            assert field.div_elements(a, b) == expected, (a, b)


def test_positive_powers_and_nonzero_power_zero(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits):
        value = 1
        if a:
            assert field.pow_element(a, 0) == 1
        for exponent in range(1, 2 * len(bits)):
            value = multiply_bits(value, value_a, modulus)
            assert field.pow_element(a, exponent) == labels[value], (a, exponent)


def test_nonzero_negative_powers(field_case):
    field, modulus, bits, labels = field_case
    for a, value_a in enumerate(bits[1:], start=1):
        inverse = next(v for v in bits[1:] if multiply_bits(value_a, v, modulus) == 1)
        value = 1
        for exponent in range(1, len(bits)):
            value = multiply_bits(value, inverse, modulus)
            assert field.pow_element(a, -exponent) == labels[value], (a, -exponent)


def test_division_requiring_exponent_minus_one(field_case):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.div_elements(1, 1) == 1
        return
    assert field.div_elements(1, 2) == len(bits) - 1


def test_alpha_inverse_power(field_case):
    field, _, bits, _ = field_case
    if len(bits) == 2:
        assert field.pow_element(1, -1) == 1
        return
    assert field.pow_element(2, -1) == len(bits) - 1


def test_negative_exponent_conversion(field_case):
    field, _, bits, _ = field_case
    period = len(bits) - 1
    for exponent in range(-3 * period, 0):
        assert field.element_from_exp(exponent) == 1 + exponent % period


def test_division_by_zero_raises(field_case):
    field, _, bits, _ = field_case
    for a in range(len(bits)):
        with pytest.raises(ZeroDivisionError):
            field.div_elements(a, 0)


def test_zero_power_zero_is_one(field_case):
    field, _, _, _ = field_case
    assert field.pow_element(0, 0) == 1


def test_zero_negative_power_raises(field_case):
    field, _, _, _ = field_case
    with pytest.raises(ZeroDivisionError):
        field.pow_element(0, -1)


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
            expected_sum = [(a[i] if i < len(a) else 0) ^ (b[i] if i < len(b) else 0) for i in range(max(len(a), len(b)))]
            assert as_polynomial_bits(field.add_poly(operand_a, operand_b), bits) == trim_polynomial(expected_sum)
            assert as_polynomial_bits(field.mult_poly(operand_a, operand_b), bits) == reference_poly_product(a, b, modulus)
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
            q, r = field.divmod_poly(operand_a, operand_b)
            assert as_polynomial_bits(q, bits) == expected_q
            assert as_polynomial_bits(r, bits) == expected_r
            assert as_polynomial_bits(field.div_poly(operand_a, operand_b), bits) == expected_q
            assert as_polynomial_bits(field.mod_poly(operand_a, operand_b), bits) == expected_r
            assert field.is_factor(operand_a, operand_b) == (expected_r == [0])
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
            assert field.substitute_element_into_poly(p, a) == labels[expected]
        np.testing.assert_array_equal(p, before)


def test_monic_multiplier(field_case):
    field, modulus, bits, labels = field_case
    coefficient = bits[-1]
    inverse = next(value for value in bits[1:] if multiply_bits(coefficient, value, modulus) == 1)
    assert field.monic_multiplier(np.array([1, labels[coefficient]])) == labels[inverse]


def test_monic_multiplier_with_trailing_zeros(field_case):
    field, modulus, bits, labels = field_case
    coefficient = bits[-1]
    inverse = next(value for value in bits[1:] if multiply_bits(coefficient, value, modulus) == 1)
    assert field.monic_multiplier(np.array([1, labels[coefficient], 0])) == labels[inverse]


def test_zero_polynomial_has_no_monic_multiplier(field_case):
    field, _, _, _ = field_case
    with pytest.raises(ZeroDivisionError):
        field.monic_multiplier(np.array([0, 0]))


def test_derivative_preserves_extension_field_coefficients(field_case):
    field, _, bits, _ = field_case
    coefficient = len(bits) - 1
    p = np.array([1, coefficient, 1, coefficient, 1])
    before = p.copy()
    np.testing.assert_array_equal(field.derive_poly(p), [coefficient, 0, coefficient, 0])
    np.testing.assert_array_equal(p, before)


def test_zero_quotient_has_integer_dtype(field_case):
    field, _, _, _ = field_case
    q, _ = field.divmod_poly(np.array([1]), np.array([1, 1]))
    assert np.issubdtype(q.dtype, np.integer)


@pytest.mark.parametrize("coefficients", [[0], [1], [0, 1], [1, 1, 1, 1]])
def test_formal_polynomial_derivative(field_case, coefficients):
    field, _, _, _ = field_case
    p = np.array(coefficients)
    expected = [p[i] if i % 2 else 0 for i in range(1, len(p))]
    np.testing.assert_array_equal(trim_polynomial(field.derive_poly(p)), trim_polynomial(expected))
    assert np.issubdtype(field.derive_poly(p).dtype, np.integer)


@pytest.mark.parametrize("method, expected", [("add_poly", [0, 1]), ("mult_poly", [1, 1])])
def test_scalar_second_polynomial_operand(field_case, method, expected):
    field, _, _, _ = field_case
    np.testing.assert_array_equal(getattr(field, method)(np.array([1, 1]), 1), expected)


@pytest.mark.parametrize("method, expected", [("add_poly", [0, 1]), ("mult_poly", [1, 1])])
def test_scalar_first_polynomial_operand(field_case, method, expected):
    field, _, _, _ = field_case
    np.testing.assert_array_equal(getattr(field, method)(1, np.array([1, 1])), expected)


def test_polynomial_division_by_zero_raises(field_case):
    field, _, _, _ = field_case
    for divisor in [0, np.array([0]), np.array([0, 0, 0])]:
        with pytest.raises(ZeroDivisionError):
            field.divmod_poly(np.array([1, 1]), divisor)


@pytest.mark.parametrize("dividend", [[0], [1, 0, 1]])
@pytest.mark.parametrize("scalar_divisor", [False, True])
def test_constant_polynomial_division_terminates(field_case, dividend, scalar_divisor):
    # Run in a child process so a nonterminating algorithm cannot hang pytest.
    _, modulus, bits, labels = field_case
    p = [(modulus >> i) & 1 for i in range(modulus.bit_length())]
    divisor_label = len(bits) - 1
    divisor_expression = str(divisor_label) if scalar_divisor else f"np.array([{divisor_label}])"
    script = (
        "import json; import numpy as np; from coding_theory_algorithms.galois_field import GaloisField; "
        f"field = GaloisField(np.array({p!r})); "
        f"q, r = field.divmod_poly(np.array({dividend!r}), {divisor_expression}); "
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
    expected_q, expected_r = reference_poly_division(dividend, [bits[divisor_label]], modulus)
    assert trim_polynomial(q) == [labels[value] for value in expected_q]
    assert trim_polynomial(r) == [labels[value] for value in expected_r]
