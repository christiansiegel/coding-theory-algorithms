"""Independent arithmetic on binary polynomial values for test expectations."""


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
    inverse = next(value for value in range(1, 1 << (modulus.bit_length() - 1)) if multiply_bits(divisor[-1], value, modulus) == 1)
    quotient = [0] * max(1, len(residual) - len(divisor) + 1)
    while residual != [0] and len(residual) >= len(divisor):
        offset = len(residual) - len(divisor)
        factor = multiply_bits(residual[-1], inverse, modulus)
        quotient[offset] ^= factor
        for i, value in enumerate(divisor):
            residual[offset + i] ^= multiply_bits(factor, value, modulus)
        residual = trim_polynomial(residual)
    return trim_polynomial(quotient), residual
