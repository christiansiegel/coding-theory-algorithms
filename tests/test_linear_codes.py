"""Small binary codes checked by enumeration and fixed textbook matrices."""

from collections import Counter
from itertools import product

import numpy as np
import pytest

from LinearBlockCode import GtoH, HtoG, LinearBlockCode


@pytest.fixture(params=["repetition3", "repetition5", "hamming7"])
def linear_case(request):
    if request.param == "hamming7":
        parity = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        generator = np.column_stack([parity, np.eye(4, dtype=int)])
        check = np.column_stack([np.eye(3, dtype=int), parity.T])
        minimum_distance = 3
    else:
        n = 3 if request.param == "repetition3" else 5
        generator = np.ones((1, n), dtype=int)
        check = np.column_stack([np.eye(n - 1, dtype=int), np.ones(n - 1, dtype=int)])
        minimum_distance = n
    code = LinearBlockCode()
    code.setG(generator)
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
    from_check.setH(check)
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
    assert code.errorDetectionCapability() == minimum_distance - 1
    assert code.Ai(0) == histogram[0] == 1
    # A() preserves the README's historical layout: weights 1 through n.
    np.testing.assert_array_equal(code.A(), [histogram[i] for i in range(1, code.n() + 1)])


def test_all_correctable_errors_and_decoding(linear_case):
    code, generator, check, minimum_distance = linear_case
    radius = (minimum_distance - 1) // 2
    errors = [np.array(e) for e in product([0, 1], repeat=code.n()) if sum(e) <= radius]
    assert {tuple(e) for e in code.correctableErrorPatterns()} == {tuple(e) for e in errors}
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
            np.testing.assert_array_equal(code.syndromeDecode(received), codeword)


def test_decoder_rejects_a_syndrome_missing_from_the_table():
    code = LinearBlockCode()
    # The README (7,3) code covers only 8 of its 16 syndromes.
    code.setG(np.array([[1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]]))
    received = np.array([1, 1, 0, 0, 0, 0, 0])
    with pytest.raises(KeyError):
        code.syndromeDecode(received)


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
            assert code.printErrorsThatHaveSyndrome(requested) is None
            records = printed_error_records(capsys.readouterr().out)
            assert len(records) == len(expected)
            assert set(records) == expected
            np.testing.assert_array_equal(requested, before)


@pytest.mark.parametrize("linear_case", ["hamming7"], indirect=True)
def test_printed_errors_for_historical_hardcoded_syndrome(linear_case, capsys):
    code, _, check, minimum_distance = linear_case
    requested = [0, 0, 1]
    assert code.printErrorsThatHaveSyndrome(requested) is None
    records = printed_error_records(capsys.readouterr().out)
    assert set(records) == reference_errors_for_syndrome(check, requested)
    assert len(records) == len(set(records)) == 16
    # A diagnostic must include errors beyond the decoder's correction radius.
    assert any(weight > (minimum_distance - 1) // 2 for _, weight in records)
    assert requested == [0, 0, 1]
