"""Check channel calculations against independent closed-form answers."""

import math

import numpy as np
import pytest

import coding_theory_algorithms.binary_symmetric_channel as channel_module
from coding_theory_algorithms.binary_symmetric_channel import BSC


def binary_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@pytest.mark.parametrize("p", [1e-200, 1e-320, np.nextafter(0.0, 1.0)])
@pytest.mark.parametrize("method", ["HX", "HY", "HXY", "HYX", "channel_efficiency"])
def test_entropy_with_rare_positive_probabilities(p, method, capsys):
    if method == "HXY":
        source, transition = [p, 1.0], [[1.0], [1.0]]
    elif method == "HYX":
        source, transition = [1.0], [[p, 1.0]]
    else:
        source, transition = [p, 1.0], [[1, 0], [0, 1]]
    channel = BSC(source, transition)
    expected = -p * math.log2(p)
    for verbose in [False, True]:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            actual = getattr(channel, method)(verbose)
        assert math.isfinite(actual) and actual > 0
        assert actual == pytest.approx(expected, rel=1e-12, abs=0)
        assert bool(capsys.readouterr().out) == verbose
        np.testing.assert_array_equal(channel.PX(), source)
        np.testing.assert_array_equal(channel.PYX(), np.asarray(transition).T)


def make_channel(source_zero_probability, crossover_probability):
    x = source_zero_probability
    p = crossover_probability
    return BSC([x, 1 - x], [[1 - p, p], [p, 1 - p]])


CONDITIONAL_CASES = [
    pytest.param([0.2, 0.8], [[0.75, 0.25], [0.25, 0.75]], [[3 / 7, 1 / 13], [4 / 7, 12 / 13]], id="biased-binary"),
    pytest.param([0.2, 0.8], [[0.5, 0.5], [0.5, 0.5]], [[0.2, 0.2], [0.8, 0.8]], id="uninformative-binary"),
    pytest.param([1, 0], [[0.75, 0.25], [0.25, 0.75]], [[1, 1], [0, 0]], id="deterministic-noisy"),
    pytest.param([0.2, 0.8], [[1, 0], [0, 1]], [[1, 0], [0, 1]], id="biased-noiseless"),
    pytest.param([1, 0], [[1, 0], [0, 1]], [[1, np.nan], [0, np.nan]], id="deterministic-noiseless"),
    pytest.param([0, 1], [[0, 1], [1, 0]], [[0, np.nan], [1, np.nan]], id="deterministic-flipping"),
    pytest.param([0.25, 0.75], [[1, 0, 0], [0, 1, 0]], [[1, 0, np.nan], [0, 1, np.nan]], id="rectangular-unreachable"),
    pytest.param([0.25, 0.75], [[0.5, 0.5, 0], [0, 0.5, 0.5]], [[1, 0.25, 0], [0, 0.75, 1]], id="rectangular-reachable"),
]


def reference_entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


@pytest.mark.parametrize("source,transition,posterior", CONDITIONAL_CASES)
def test_posterior_known_answers(source, transition, posterior, capsys):
    channel = BSC(source, transition)
    for verbose in [False, True]:
        with np.errstate(divide="raise", invalid="raise"):
            actual = channel.PXY(verbose)
        np.testing.assert_allclose(actual, posterior, equal_nan=True)
        assert actual.shape == np.asarray(transition).shape
        assert np.issubdtype(actual.dtype, np.floating)
        reachable = ~np.isnan(np.asarray(posterior)).all(axis=0)
        np.testing.assert_allclose(actual[:, reachable].sum(axis=0), 1)
        output = capsys.readouterr().out
        assert bool(output) == verbose
        np.testing.assert_array_equal(channel.PX(), source)
        np.testing.assert_array_equal(channel.PYX(), np.asarray(transition).T)


@pytest.mark.parametrize("source,transition,posterior", CONDITIONAL_CASES)
def test_conditional_entropy_from_joint_distribution(source, transition, posterior, capsys):
    # Compute H(X,Y)-H(Y) with ordinary scalar arithmetic, independently of
    # the production posterior and entropy methods.
    joint = [[px * probability for probability in row] for px, row in zip(source, transition)]
    output_probabilities = [sum(column) for column in zip(*joint)]
    expected = reference_entropy([p for row in joint for p in row]) - reference_entropy(output_probabilities)
    channel = BSC(source, transition)
    for verbose in [False, True]:
        with np.errstate(divide="raise", invalid="raise"):
            actual = channel.HXY(verbose)
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected, abs=1e-12)
        assert channel.HX() + channel.HYX() == pytest.approx(channel.HY() + actual, abs=1e-12)
        output = capsys.readouterr().out
        assert bool(output) == verbose
        np.testing.assert_array_equal(channel.PX(), source)
        np.testing.assert_array_equal(channel.PYX(), np.asarray(transition).T)


@pytest.mark.parametrize("p", [0, 0.1, 0.25, 0.5, 0.9, 1])
def test_capacity_known_formula_and_source_preservation(p, capsys):
    channel = make_channel(0.2, p)
    before = channel.PX().copy()
    assert channel.Cs(False) == pytest.approx(1 - binary_entropy(p))
    np.testing.assert_array_equal(channel.PX(), before)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("x", [0, 0.2, 0.5, 1])
@pytest.mark.parametrize("p", [0, 0.1, 0.25, 0.9, 1])
def test_efficiency_known_formula(x, p, capsys):
    channel = make_channel(x, p)
    output_zero_probability = x * (1 - p) + (1 - x) * p
    expected = (binary_entropy(output_zero_probability) - binary_entropy(p)) / (1 - binary_entropy(p))
    np.testing.assert_allclose(channel.PY(), [output_zero_probability, 1 - output_zero_probability])
    assert channel.HY() == pytest.approx(binary_entropy(output_zero_probability))
    assert channel.HYX() == pytest.approx(binary_entropy(p))
    before = channel.PX().copy()
    assert channel.channel_efficiency(False) == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(channel.PX(), before)
    assert capsys.readouterr().out == ""


def test_efficiency_uses_its_own_instance(monkeypatch):
    # A module-global example must not affect another channel's calculation.
    monkeypatch.setattr(channel_module, "bsc", make_channel(0.5, 0.1), raising=False)
    channel = make_channel(0.2, 0.1)
    expected = (binary_entropy(0.26) - binary_entropy(0.1)) / (1 - binary_entropy(0.1))
    assert channel.channel_efficiency(False) == pytest.approx(expected)


def test_capacity_failure_does_not_change_source(monkeypatch):
    channel = make_channel(0.2, 0.1)
    before = channel.PX().copy()

    def fail_log(*args, **kwargs):
        raise RuntimeError("capacity calculation failed")

    monkeypatch.setattr(channel_module.np, "log", fail_log)
    with pytest.raises(RuntimeError, match="capacity calculation failed"):
        channel.Cs(False)
    np.testing.assert_array_equal(channel.PX(), before)


def test_efficiency_at_zero_capacity_raises():
    channel = make_channel(0.2, 0.5)
    with pytest.raises(ZeroDivisionError):
        channel.channel_efficiency(False)


def test_verbose_efficiency_includes_the_result(capsys):
    channel = make_channel(0.5, 0.25)
    assert channel.channel_efficiency(True) == pytest.approx(1)
    assert "η = I(X,Y) / Cs =" in capsys.readouterr().out


@pytest.mark.parametrize("x", [0, 0.2, 0.5, 1])
def test_source_entropy_including_zero_probabilities(x):
    channel = make_channel(x, 0.1)
    assert channel.HX() == pytest.approx(binary_entropy(x))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.8])
def test_asymmetric_z_channel_capacity(p, capsys):
    # For P=[[1,0],[p,1-p]], maximize h(u*(1-p))-u*h(p), where
    # u=P(X=1). Setting its derivative to zero gives this closed form.
    expected = math.log2(1 + (1 - p) * p ** (p / (1 - p)))
    channel = BSC([0.2, 0.8], [[1, 0], [p, 1 - p]])
    before = channel.PX().copy()
    assert channel.Cs(False) == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(channel.PX(), before)
    assert capsys.readouterr().out == ""


def test_asymmetric_channel_efficiency_at_optimal_source(capsys):
    # For the p=1/2 Z channel, the maximizing source is [3/5,2/5].
    channel = BSC([0.6, 0.4], [[1, 0], [0.5, 0.5]])
    assert channel.HY() - channel.HYX() == pytest.approx(math.log2(1.25))
    assert channel.channel_efficiency(False) == pytest.approx(1, abs=1e-10)
    np.testing.assert_array_equal(channel.PX(), [0.6, 0.4])
    assert capsys.readouterr().out == ""


def test_capacity_with_duplicate_input_rows():
    # Inputs 0 and 1 have the same output; only two symbols can be distinguished.
    channel = BSC([0.2, 0.3, 0.5], [[1, 0], [1, 0], [0, 1]])
    assert channel.Cs(False) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize(
    "source, transition, expected",
    [
        ([0.2, 0.8], [[1, 0, 0], [0, 1, 0]], 1),
        ([0.2, 0.3, 0.5], np.eye(3), math.log2(3)),
        ([0.2, 0.8], [[0.25, 0.75], [0.25, 0.75]], 0),
        ([0.2, 0.8], [[1, 0, 0], [1, 0, 0]], 0),
    ],
)
def test_general_capacity_boundary_cases(source, transition, expected, capsys):
    channel = BSC(source, transition)
    with np.errstate(divide="raise", invalid="raise"):
        assert channel.Cs(False) == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(channel.PX(), source)
    assert capsys.readouterr().out == ""


def test_capacity_with_an_unhelpful_input_and_permuted_symbols():
    # The third row adds pure noise to an otherwise binary symmetric channel.
    transition = np.array([[0.75, 0.25, 0], [0.25, 0.75, 0], [0.5, 0.5, 0]])
    expected = 1 - binary_entropy(0.25)
    for matrix in [transition, transition[[2, 0, 1]][:, [2, 1, 0]]]:
        channel = BSC([0.2, 0.3, 0.5], matrix)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            assert channel.Cs(False) == pytest.approx(expected, abs=1e-10)


def test_verbose_asymmetric_capacity_includes_the_result(capsys):
    channel = BSC([0.2, 0.8], [[1, 0], [0.5, 0.5]])
    assert channel.Cs(True) == pytest.approx(math.log2(1.25), abs=1e-10)
    assert "Cs = Imax(X,Y) =" in capsys.readouterr().out
    np.testing.assert_array_equal(channel.PX(), [0.2, 0.8])


def test_capacity_iteration_limit_raises_and_preserves_source(monkeypatch):
    channel = BSC([0.2, 0.8], [[1, 0], [0.5, 0.5]])
    monkeypatch.setattr(channel_module, "_CAPACITY_MAX_ITERATIONS", 1)
    with pytest.raises(RuntimeError, match="did not converge"):
        channel.Cs(False)
    np.testing.assert_array_equal(channel.PX(), [0.2, 0.8])


def test_capacity_for_the_script_example_with_boundary_optimum(capsys):
    # H(Y|X)=log2(3) for every input. H(Y) is maximized by the output
    # [1/6,1/6,1/3,1/6,1/6], attained by source [1/2,0,1/2].
    # Therefore capacity is exactly 2/3 bits.
    channel = BSC(
        [1 / 3] * 3,
        [
            [1 / 3, 1 / 3, 1 / 3, 0, 0],
            [0, 1 / 3, 1 / 3, 1 / 3, 0],
            [0, 0, 1 / 3, 1 / 3, 1 / 3],
        ],
    )
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        assert channel.Cs(False) == pytest.approx(2 / 3, abs=1e-10)
    np.testing.assert_array_equal(channel.PX(), [1 / 3] * 3)
    assert capsys.readouterr().out == ""
