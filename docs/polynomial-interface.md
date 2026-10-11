# Polynomial arithmetic over GF(2^m)

Polynomials here have **field-element labels as coefficients**, not binary
polynomial values. See the [element contract](finite-field-interface.md) for
that distinction. Roots, minimal
polynomials, irreducibility tests, and polynomial GCD are outside this step.

## Representation

Use a one-dimensional integer NumPy array `[a0, a1, ..., ad]` for
a0 + a1 X + ... + ad X^d. Every coefficient is a canonical label in `[0, q)`.
Trailing zeros are allowed and do not change the polynomial's value. Input
arrays must contain at least one coefficient. Results are integer arrays;
zero results can be represented as `[0]`, a padded zero vector, or
an empty integer array for compatibility with the existing API. Tests compare
values independently of zero padding.

`addPoly`, `multPoly`, and `divmodPoly` also accept Python integer labels as
constant polynomials in either operand. Input arrays must not be mutated.
Validation of noninteger coefficients and out-of-range labels is not yet
established. Malformed or empty inputs have no established validation behavior.

## Operations

| Method | Contract |
| --- | --- |
| `addPoly(a, b)` | Coefficient-wise field addition; subtraction is identical in characteristic two. |
| `multPoly(a, b)` | Polynomial convolution with field multiplication and addition. |
| `divmodPoly(a, b, verbose=False)` | Return `[quotient, remainder]` with a = b*quotient + remainder, and remainder zero or degree less than b. Must terminate for any nonzero divisor, including a constant. A zero divisor raises `ZeroDivisionError`. |
| `divPoly(a, b)`, `modPoly(a, b)` | Return the quotient or remainder of the same division. |
| `isFactor(p, factorPoly)` | Return whether the remainder is zero; a zero factor raises `ZeroDivisionError`. |
| `derivePoly(p)` | Formal derivative: coefficient i contributes i*a_i at position i-1. In characteristic two this is a_i for odd i, zero for even i. Constants differentiate to zero. |
| `substituteElementIntoPoly(p, a)` | Evaluate p at field label a and return the resulting integer label; evaluation at zero returns the constant coefficient. |
| `monicMultiplier(p)` | Return the inverse of the highest-degree nonzero coefficient, ignoring trailing zeros. Multiplying every coefficient by it makes the polynomial monic. A zero polynomial raises `ZeroDivisionError`. |

The formal derivative multiplies coefficients by integer degree; it does
**not** raise them to that degree.

## Verification

Tests reuse the independent binary-polynomial element reference in GF(2),
GF(4), GF(8), and GF(16). Polynomial multiplication and long division run on
binary field values rather than production labels. Evaluation uses an
independent Horner calculation. Selected inputs include zero polynomials,
non-monic divisors, extension-field coefficients, and trailing zeros.

Regressions cover scalar operands, derivatives,
division by zero, integer zero quotients, and monic multipliers with
trailing zeros. Constant-divisor tests run in subprocesses with three-second
timeouts to guard against the original nonterminating loop. They cover both
scalar and array constant divisors across the four fields. Timeouts, crashes,
or incorrect results fail the tests.
