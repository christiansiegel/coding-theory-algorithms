"""Known BCH/RS answers built from independent field-polynomial arithmetic."""

from itertools import product

import numpy as np
import pytest

from BCHCode import BCHCode
from GaloisField import GaloisField
from RSCode import RSCode

from .field_reference import (
    multiply_bits,
    reference_poly_division,
    reference_poly_product,
)


def field_reference(modulus):
    m = modulus.bit_length() - 1
    field = GaloisField(np.array([(modulus >> i) & 1 for i in range(m + 1)]))
    bits = [0]
    value = 1
    for _ in range((1 << m) - 1):
        bits.append(value)
        value = multiply_bits(value, 2, modulus)
    assert value == 1
    assert len(set(bits)) == 1 << m
    return field, bits, {value: label for label, value in enumerate(bits)}


def generator_from_roots(modulus, bits, t, binary):
    n = len(bits) - 1
    exponents = set(range(1, 2 * t + 1))
    if binary:
        # Binary generators include every Frobenius conjugate, exactly once.
        for exponent in list(exponents):
            conjugate = exponent % n
            while True:
                exponents.add(conjugate)
                conjugate = 2 * conjugate % n
                if conjugate == exponent % n:
                    break
    generator = [1]
    for exponent in sorted(exponents):
        root = bits[1 + exponent % n]
        generator = reference_poly_product(generator, [root, 1], modulus)
    return generator


def reference_encode(message, generator, modulus, n, systematic=True):
    if systematic:
        shifted = [0] * (len(generator) - 1) + list(message)
        _, residual = reference_poly_division(shifted, generator, modulus)
        result = shifted[:]
        for i, coefficient in enumerate(residual):
            result[i] ^= coefficient
    else:
        result = reference_poly_product(message, generator, modulus)
    return result + [0] * (n - len(result))


def reference_syndrome(received, bits, modulus, count):
    result = []
    for exponent in range(1, count + 1):
        root = bits[1 + exponent % (len(bits) - 1)]
        value = 0
        for coefficient in reversed(received):
            value = multiply_bits(value, root, modulus) ^ coefficient
        result.append(value)
    return result


def padded_bits(decoded, bits, n):
    assert len(decoded) <= n
    return [bits[int(label)] for label in decoded] + [0] * (n - len(decoded))


@pytest.mark.parametrize("modulus, t", [(0b1011, 1), (0b10011, 2)])
def test_bch_generator_and_all_encodings(modulus, t, capsys):
    field, bits, _ = field_reference(modulus)
    code = BCHCode(field, t)
    expected_generator = generator_from_roots(modulus, bits, t, binary=True)
    assert set(expected_generator) <= {0, 1}
    np.testing.assert_array_equal(code.g(), expected_generator)
    assert code.n() == len(bits) - 1
    assert code.k() == code.n() - len(expected_generator) + 1
    assert code.t() == t
    for message in product([0, 1], repeat=code.k()):
        for systematic in [False, True]:
            expected = reference_encode(
                message, expected_generator, modulus, code.n(), systematic
            )
            np.testing.assert_array_equal(
                code.c(np.array(message), systematic), expected
            )
    assert capsys.readouterr().out == ""


def test_bch_syndromes_for_all_received_vectors(capsys):
    modulus = 0b1011
    field, bits, labels = field_reference(modulus)
    code = BCHCode(field, 1)
    for received in product([0, 1], repeat=7):
        expected = reference_syndrome(received, bits, modulus, 2)
        np.testing.assert_array_equal(
            code.S(np.array(received), False), [labels[v] for v in expected]
        )
    assert capsys.readouterr().out == ""


def test_bch_corrects_single_errors_at_every_position(capsys):
    modulus = 0b1011
    field, bits, _ = field_reference(modulus)
    code = BCHCode(field, 1)
    generator = generator_from_roots(modulus, bits, 1, binary=True)
    for message in [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]]:
        codeword = reference_encode(message, generator, modulus, 7)
        for position in range(7):
            received = codeword[:]
            received[position] ^= 1
            decoded = code.decode(np.array(received), False)
            assert padded_bits(decoded, bits, 7) == codeword
    assert capsys.readouterr().out == ""


def test_bch_corrects_selected_two_error_patterns(capsys):
    field, bits, _ = field_reference(0b10011)
    code = BCHCode(field, 2)
    for positions in [(0, 1), (0, 14), (3, 7), (7, 14)]:
        received = np.zeros(15, dtype=int)
        received[list(positions)] = 1
        decoded = code.decode(received, False)
        assert padded_bits(decoded, bits, 15) == [0] * 15
    assert capsys.readouterr().out == ""


def test_bch_corrects_an_error_canceling_the_highest_term():
    modulus = 0b1011
    field, bits, _ = field_reference(modulus)
    generator = generator_from_roots(modulus, bits, 1, binary=True)
    expected = reference_encode([1, 0, 0, 0], generator, modulus, 7)
    assert expected == [1, 1, 0, 1, 0, 0, 0]
    received = expected[:]
    received[3] ^= 1
    code = BCHCode(field, 1)
    assert padded_bits(code.decode(np.array(received), False), bits, 7) == expected


@pytest.mark.parametrize("message", [[0, 0, 0, 0], [1, 0, 1, 0]])
def test_bch_decodes_error_free_codewords(message):
    modulus = 0b1011
    field, bits, _ = field_reference(modulus)
    code = BCHCode(field, 1)
    generator = generator_from_roots(modulus, bits, 1, binary=True)
    codeword = reference_encode(message, generator, modulus, 7)
    received = np.array(codeword)
    decoded = code.decode(received, False)
    assert padded_bits(decoded, bits, 7) == codeword
    decoded[0] ^= 1
    np.testing.assert_array_equal(received, codeword)


@pytest.mark.parametrize("modulus, t", [(0b1011, 1), (0b10011, 2)])
def test_bch_binary_expanded_parity_check(modulus, t):
    field, bits, _ = field_reference(modulus)
    code = BCHCode(field, t)
    m = modulus.bit_length() - 1
    n = len(bits) - 1
    expected = np.array(
        [
            [(bits[1 + exponent * j % n] >> i) & 1 for j in range(n)]
            for exponent in range(1, 2 * t, 2)
            for i in range(m)
        ]
    )
    check = code.H(False)
    np.testing.assert_array_equal(check, expected)
    np.testing.assert_array_equal(code.G() @ check.T % 2, np.zeros((code.k(), m * t)))


def test_bch_generator_deduplicates_conjugate_root_groups():
    modulus = 0b10011
    field, bits, _ = field_reference(modulus)
    expected = generator_from_roots(modulus, bits, 5, binary=True)
    assert expected == [1] * 15
    code = BCHCode(field, 5)
    np.testing.assert_array_equal(code.g(), expected)
    assert code.k() == 1


@pytest.mark.parametrize("modulus, t", [(0b111, 1), (0b1011, 1), (0b1011, 2)])
def test_rs_generator_roots_and_dimensions(modulus, t):
    field, bits, labels = field_reference(modulus)
    code = RSCode(field, t)
    expected = generator_from_roots(modulus, bits, t, binary=False)
    np.testing.assert_array_equal(code.g(), [labels[v] for v in expected])
    assert code.n() == len(bits) - 1
    assert code.k() == code.n() - 2 * t


@pytest.mark.parametrize("systematic", [False, True])
@pytest.mark.parametrize("modulus, t", [(0b111, 1), (0b1011, 2)])
def test_rs_encodes_every_message_in_small_codes(modulus, t, systematic, capsys):
    field, bits, labels = field_reference(modulus)
    code = RSCode(field, t)
    generator = generator_from_roots(modulus, bits, t, binary=False)
    minimum_weight = code.n()
    for message in product(range(len(bits)), repeat=code.k()):
        expected = reference_encode(
            [bits[v] for v in message], generator, modulus, code.n(), systematic
        )
        np.testing.assert_array_equal(
            code.c(np.array(message), systematic), [labels[v] for v in expected]
        )
        if systematic:
            assert expected[-code.k() :] == [bits[v] for v in message]
        weight = sum(v != 0 for v in expected)
        if weight:
            minimum_weight = min(minimum_weight, weight)
    assert minimum_weight == 2 * t + 1
    assert code.dmin() == minimum_weight
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("method, expected", [("dmin", 3), ("t", 1)])
def test_rs_distance_and_correction_radius_over_gf8(method, expected):
    field, _, _ = field_reference(0b1011)
    assert getattr(RSCode(field, 1), method)() == expected


def test_rs_syndromes_for_all_gf4_received_vectors():
    modulus = 0b111
    field, bits, labels = field_reference(modulus)
    code = RSCode(field, 1)
    for received in product(range(4), repeat=3):
        expected = reference_syndrome([bits[v] for v in received], bits, modulus, 2)
        np.testing.assert_array_equal(
            code.S(np.array(received), False), [labels[v] for v in expected]
        )


def test_rs_corrects_all_single_symbol_errors_over_gf4():
    modulus = 0b111
    field, bits, labels = field_reference(modulus)
    code = RSCode(field, 1)
    for value in bits:
        codeword = [value] * 3  # The (3,1) code is a repetition code over GF(4).
        for position in range(3):
            for error in bits[1:]:
                received = codeword[:]
                received[position] ^= error
                decoded = code.decode(np.array([labels[v] for v in received]), False)
                assert padded_bits(decoded, bits, 3) == codeword


def test_rs_corrects_an_error_canceling_the_highest_term():
    field, bits, _ = field_reference(0b111)
    code = RSCode(field, 1)
    assert (
        padded_bits(code.decode(np.array([2, 2, 0]), False), bits, 3) == [bits[2]] * 3
    )


def test_rs_decodes_an_error_free_codeword():
    field, bits, _ = field_reference(0b111)
    code = RSCode(field, 1)
    np.testing.assert_array_equal(
        padded_bits(code.decode(np.array([2, 2, 2]), False), bits, 3), [bits[2]] * 3
    )


@pytest.mark.parametrize("t", [1, 2])
def test_rs_gf8_syndromes_and_decoding_with_nonbinary_symbols(t, capsys):
    modulus = 0b1011
    field, bits, labels = field_reference(modulus)
    code = RSCode(field, t)
    generator = generator_from_roots(modulus, bits, t, binary=False)
    for message in [[0] * code.k(), [2] * code.k(), list(range(2, 2 + code.k()))]:
        expected = reference_encode([bits[v] for v in message], generator, modulus, 7)
        for systematic in [False, True]:
            np.testing.assert_array_equal(
                code.c(np.array(message), systematic),
                [
                    labels[v]
                    for v in reference_encode(
                        [bits[v] for v in message], generator, modulus, 7, systematic
                    )
                ],
            )
        # Include zero errors and the boundary positions, with distinct magnitudes.
        patterns = [[], [(0, 2)], [(6, 5)], [(3, 7)]]
        if t == 2:
            patterns += [[(0, 2), (6, 5)], [(2, 7), (5, 3)]]
        for errors in patterns:
            received = expected[:]
            for position, error in errors:
                received[position] ^= bits[error]
            received_labels = np.array([labels[v] for v in received])
            syndrome = reference_syndrome(received, bits, modulus, 2 * t)
            np.testing.assert_array_equal(
                code.S(received_labels, False), [labels[v] for v in syndrome]
            )
            assert padded_bits(code.decode(received_labels, False), bits, 7) == expected
    assert capsys.readouterr().out == ""
