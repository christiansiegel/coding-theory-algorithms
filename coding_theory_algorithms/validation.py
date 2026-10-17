"""Validation at public construction and encoding boundaries."""

from numbers import Integral, Real

import numpy as np


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def probability(value):
    if not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("p must be a finite probability in [0, 1]")
    return float(value)


def coefficients(value, name, alphabet=2, length=None, maximum_length=None):
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector of integer coefficients")
    if np.any(array != np.floor(array)) or np.any(array < 0) or np.any(array >= alphabet):
        raise ValueError(f"{name} coefficients must be integers in [0, {alphabet})")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}")
    if maximum_length is not None and array.size > maximum_length:
        raise ValueError(f"{name} must have at most {maximum_length} coefficients")
    return array.astype(int, copy=True)


def binary_matrix(value, name, generator):
    array = np.asarray(value)
    if array.ndim != 2 or not 0 < array.shape[0] < array.shape[1]:
        raise ValueError(f"{name} must have shape (rows, columns) with 0 < rows < columns")
    result = coefficients(array.reshape(-1), name).reshape(array.shape)
    rows = result.shape[0]
    identity = result[:, -rows:] if generator else result[:, :rows]
    if not np.array_equal(identity, np.eye(rows, dtype=int)):
        raise ValueError(f"{name} must use systematic layout {'[P | I]' if generator else '[I | P.T]'}")
    return result


def binary_remainder(value, divisor):
    while value and value.bit_length() >= divisor.bit_length():
        value ^= divisor << (value.bit_length() - divisor.bit_length())
    return value


def primitive_polynomial(value):
    polynomial = coefficients(value, "p")
    while polynomial.size > 1 and polynomial[-1] == 0:
        polynomial = polynomial[:-1]
    if polynomial.size < 2 or polynomial[0] != 1:
        raise ValueError("p must be a positive-degree primitive binary polynomial")
    mask = sum(int(bit) << i for i, bit in enumerate(polynomial))
    order = (1 << (polynomial.size - 1)) - 1
    element = 1
    for exponent in range(1, order + 1):
        element = binary_remainder(element << 1, mask)
        if element in (0, 1):
            if element == 1 and exponent == order:
                return polynomial.copy()
            break
    raise ValueError("p must be primitive over GF(2)")
