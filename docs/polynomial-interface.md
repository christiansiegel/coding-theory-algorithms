# Polynomial arithmetic over GF(2^m)

Polynomials here have **field-element labels as coefficients**, not binary
polynomial values. See the [element contract](finite-field-interface.md) for
that distinction. This contract records intended mathematical behavior;
strict expected-failure tests mark current departures. Roots, minimal
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
| `monicMultiplier(p)` | For a nonzero polynomial, return the inverse of its highest-degree nonzero coefficient, ignoring trailing zeros. Multiplying every coefficient by it makes the polynomial monic. |

The formal derivative multiplies coefficients by integer degree; it does
**not** raise them to that degree. Handling the zero polynomial in
`monicMultiplier()` is not established in this step.

## Verification and known failures

Tests reuse the independent binary-polynomial element reference in GF(2),
GF(4), GF(8), and GF(16). Polynomial multiplication and long division run on
binary field values rather than production labels. Evaluation uses an
independent Horner calculation. Selected inputs include zero polynomials,
non-monic divisors, extension-field coefficients, and trailing zeros.

Expected-failure regressions cover scalar second operands, derivatives,
division by zero, a floating-point zero quotient, and monic multipliers with
trailing zeros. Constant-divisor tests run in subprocesses with three-second
timeouts because the current division loop does not terminate. A timeout is
the only accepted expected failure in those tests; crashes or incorrect results
remain failures.
