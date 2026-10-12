"""Small binary codes checked by enumeration and fixed textbook matrices."""

from collections import Counter
from itertools import product
import math

import numpy as np
import pytest

from coding_theory_algorithms.linear_block_code import GtoH, HtoG, LinearBlockCode


@pytest.fixture(params=["repetition3", "repetition5", "hamming7"])
def linear_case(request):
    if request.param == "hamming7":
        parity = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        generator = np.column_stack([parity, np.eye(4, dtype=int)])
        check = np.column_stack([np.eye(3, dtype=int), parity.T])
        minimum_distance = 3
    elif request.param == "readme7":
        generator = np.array([[1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
        check = np.column_stack([np.eye(4, dtype=int), generator[:, :4].T])
        minimum_distance = 4
    else:
        n = 3 if request.param == "repetition3" else 5
        generator = np.ones((1, n), dtype=int)
        check = np.column_stack([np.eye(n - 1, dtype=int), np.ones(n - 1, dtype=int)])
        minimum_distance = n
    code = LinearBlockCode()
    code.set_g(generator)
    return code, generator, check, minimum_distance


def reference_encode(message, generator):
    # Each output bit is the parity of the selected generator-column entries.
    return np.array([sum(bit * row[j] for bit, row in zip(message, generator)) % 2 for j in range(generator.shape[1])])


def test_systematic_matrix_conversion(linear_case):
    code, generator, check, _ = linear_case
    np.testing.assert_array_equal(GtoH(generator), check)
    np.testing.assert_array_equal(HtoG(check), generator)
    np.testing.assert_array_equal(code.H(), check)
    np.testing.assert_array_equal(generator @ check.T % 2, 0)
    from_check = LinearBlockCode()
    from_check.set_h(check)
    np.testing.assert_array_equal(from_check.G(), generator)


def test_encoding_and_enumeration(linear_case):
    code, generator, _, _ = linear_case
    expected_messages = {tuple(message) for message in product([0, 1], repeat=generator.shape[0])}
    expected_codewords = set()
    for message in expected_messages:
        expected = reference_encode(message, generator)
        actual = code.c(np.array(message))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[-code.k() :], message)
        np.testing.assert_array_equal(code.s(actual), np.zeros(code.n() - code.k()))
        expected_codewords.add(tuple(expected))
    assert code.n() == generator.shape[1]
    assert code.k() == generator.shape[0]
    assert code.R() == pytest.approx(code.k() / code.n())
    assert {tuple(row) for row in code.M()} == expected_messages
    assert {tuple(row) for row in code.C()} == expected_codewords
    assert len(expected_codewords) == 2 ** code.k()
    messages_in_order = [[(i >> j) & 1 for j in range(code.k())] for i in range(2 ** code.k())]
    np.testing.assert_array_equal(code.M(), messages_in_order)
    np.testing.assert_array_equal(code.C(), [reference_encode(m, generator) for m in messages_in_order])


def test_distance_and_weight_distribution(linear_case):
    code, generator, _, minimum_distance = linear_case
    histogram = Counter(int(sum(reference_encode(message, generator))) for message in product([0, 1], repeat=code.k()))
    assert code.dmin() == minimum_distance
    assert code.t() == (minimum_distance - 1) // 2
    assert code.error_detection_capability() == minimum_distance - 1
    assert code.Ai(0) == histogram[0] == 1
    # A() preserves the README's historical layout: weights 1 through n.
    np.testing.assert_array_equal(code.A(), [histogram[i] for i in range(1, code.n() + 1)])


def test_all_correctable_errors_and_decoding(linear_case):
    code, generator, check, minimum_distance = linear_case
    radius = (minimum_distance - 1) // 2
    errors = [np.array(e) for e in product([0, 1], repeat=code.n()) if sum(e) <= radius]
    assert {tuple(e) for e in code.correctable_error_patterns()} == {tuple(e) for e in errors}
    syndromes = set()
    for error in errors:
        expected = np.array([sum(e * h for e, h in zip(error, row)) % 2 for row in check])
        np.testing.assert_array_equal(code.s(error), expected)
        syndromes.add(tuple(expected))
    assert len(syndromes) == len(errors)
    for message in product([0, 1], repeat=code.k()):
        codeword = reference_encode(message, generator)
        for error in errors:
            received = (codeword + error) % 2
            np.testing.assert_array_equal(code.syndrome_decode(received), codeword)


def test_decoder_rejects_a_syndrome_missing_from_the_table():
    code = LinearBlockCode()
    # This nonperfect (7,3) code covers only 8 of its 16 syndromes.
    code.set_g(np.array([[1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]]))
    received = np.array([1, 1, 0, 0, 0, 0, 0])
    with pytest.raises(KeyError):
        code.syndrome_decode(received)


@pytest.mark.parametrize("linear_case", ["repetition3", "repetition5", "hamming7", "readme7"], indirect=True)
def test_standard_array_partitions_all_vectors_with_minimum_weight_leaders(linear_case, capsys):
    code, generator, check, minimum_distance = linear_case
    n = generator.shape[1]
    k = generator.shape[0]
    codewords = [tuple(reference_encode([(i >> j) & 1 for j in range(k)], generator)) for i in range(2**k)]
    assert code.print_standard_array() is None
    rows = []
    for line in capsys.readouterr().out.splitlines():
        if line and set(line) == {"-"}:
            continue
        assert line.count("|") == 1
        row = [tuple(int(bit) for bit in word) for word in line.replace("|", " ").split()]
        assert len(row) == 2**k
        assert all(len(word) == n and set(word) <= {0, 1} for word in row)
        rows.append(row)
    assert len(rows) == 2 ** (n - k)
    assert rows[0] == codewords
    flattened = [word for row in rows for word in row]
    assert len(flattened) == len(set(flattened)) == 2**n
    assert set(flattened) == set(product([0, 1], repeat=n))
    for row in rows:
        leader = row[0]
        assert sum(leader) == min(map(sum, row))
        assert row == [tuple(a ^ b for a, b in zip(leader, word)) for word in codewords]
        syndromes = {tuple(sum(bit * coefficient for bit, coefficient in zip(word, parity_row)) % 2 for parity_row in check) for word in row}
        assert len(syndromes) == 1
    # Completing the diagnostic must not expand the guaranteed decoder table.
    radius = (minimum_distance - 1) // 2
    expected_errors = {error for error in product([0, 1], repeat=n) if sum(error) <= radius}
    assert {tuple(error) for error in code.decoding_table().values()} == expected_errors
    np.testing.assert_array_equal(code.G(), generator)


@pytest.mark.parametrize("p", [0, 0.01, 0.2, 0.5, 0.8, 1])
def test_undetected_error_probability_by_enumeration(linear_case, p, capsys):
    code, generator, check, _ = linear_case
    probabilities = []
    for error in product([0, 1], repeat=generator.shape[1]):
        if any(error) and all(sum(bit * coefficient for bit, coefficient in zip(error, row)) % 2 == 0 for row in check):
            probabilities.append(math.prod(p if bit else 1 - p for bit in error))
    actual = code.PU(p)
    assert actual == pytest.approx(math.fsum(probabilities), abs=1e-14)
    assert 0 <= actual <= 1
    np.testing.assert_array_equal(code.G(), generator)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("p", [0, 0.01, 0.2, 0.5, 0.8, 1])
def test_error_probability_outside_radius_by_enumeration(linear_case, p, capsys):
    code, generator, _, minimum_distance = linear_case
    radius = (minimum_distance - 1) // 2
    expected = math.fsum(math.prod(p if bit else 1 - p for bit in error) for error in product([0, 1], repeat=generator.shape[1]) if sum(error) > radius)
    actual = code.Pe(p)
    assert actual == pytest.approx(expected, abs=1e-14)
    assert 0 <= actual <= 1
    np.testing.assert_array_equal(code.G(), generator)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("p,expected", [(0, 0), (0.5, 0.5), (1, 1)])
def test_repetition_probability_without_factorial_overflow(p, expected):
    # Odd-length repetition has radius 200. At p=1/2, complementing an
    # error vector pairs every correctable pattern with an uncorrectable one.
    # This oracle needs neither enumeration of 2**401 vectors nor factorials.
    code = LinearBlockCode()
    code.set_g(np.ones((1, 401), dtype=int))
    assert code.Pe(p) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("length", [1100, 1101])
@pytest.mark.parametrize("numerator", [0, 1, 2, 3, 4])
def test_long_repetition_probability_against_exact_integer_weights(length, numerator, capsys):
    code = LinearBlockCode()
    generator = np.ones((1, length), dtype=int)
    code.set_g(generator)
    radius = (length - 1) // 2
    # For p=a/4, sum integer event weights first, then divide once. No
    # intermediate coefficient is converted to float, and no recurrence is used.
    total = sum(math.comb(length, i) * numerator**i * (4 - numerator) ** (length - i) for i in range(radius + 1, length + 1))
    expected = total / 4**length
    actual = code.Pe(numerator / 4)
    assert math.isfinite(actual) and 0 <= actual <= 1
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)
    np.testing.assert_array_equal(code.G(), generator)
    assert capsys.readouterr().out == ""


def test_repetition_probability_preserves_a_tiny_representable_tail():
    code = LinearBlockCode()
    code.set_g([[1, 1, 1]])
    p = 1e-160
    expected = 3 * p * p * (1 - p) + p**3
    actual = code.Pe(p)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


def printed_error_records(output):
    lines = output.splitlines()
    assert lines and "weight" in lines[0]
    records = []
    for line in lines[1:]:
        vector, weight = line.split("->")
        vector = vector.strip()
        assert vector.startswith("[") and vector.endswith("]"), line
        records.append((tuple(int(bit) for bit in vector[1:-1].split()), int(weight)))
    return records


def reference_errors_for_syndrome(check, syndrome):
    records = set()
    for error in product([0, 1], repeat=check.shape[1]):
        # Use the fixed textbook check matrix and ordinary integer parity,
        # not the production syndrome or weight functions.
        actual = [sum(bit * coefficient for bit, coefficient in zip(error, row)) % 2 for row in check]
        if actual == list(syndrome):
            records.add((error, sum(error)))
    return records


def test_printed_errors_for_every_requested_syndrome(linear_case, capsys):
    code, generator, check, _ = linear_case
    for syndrome in product([0, 1], repeat=check.shape[0]):
        expected = reference_errors_for_syndrome(check, syndrome)
        assert len(expected) == 2 ** generator.shape[0]
        for as_array in [False, True]:
            requested = np.array(syndrome) if as_array else list(syndrome)
            before = requested.copy()
            assert code.print_errors_that_have_syndrome(requested) is None
            records = printed_error_records(capsys.readouterr().out)
            assert len(records) == len(expected)
            assert set(records) == expected
            np.testing.assert_array_equal(requested, before)


@pytest.mark.parametrize("linear_case", ["hamming7"], indirect=True)
def test_printed_errors_for_historical_hardcoded_syndrome(linear_case, capsys):
    code, _, check, minimum_distance = linear_case
    requested = [0, 0, 1]
    assert code.print_errors_that_have_syndrome(requested) is None
    records = printed_error_records(capsys.readouterr().out)
    assert set(records) == reference_errors_for_syndrome(check, requested)
    assert len(records) == len(set(records)) == 16
    # A diagnostic must include errors beyond the decoder's correction radius.
    assert any(weight > (minimum_distance - 1) // 2 for _, weight in records)
    assert requested == [0, 0, 1]
