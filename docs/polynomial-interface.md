# Polynomial arithmetic over GF(2^m)

Polynomials here have **field-element labels as coefficients**, not binary
polynomial values. See the [element contract](finite-field-interface.md) for
that distinction. Roots, minimal polynomials, and binary classification have
contracts in the field document. The early-stopping decoder Euclidean helper
has a [separate contract](euclidean-interface.md); it is not a full polynomial GCD.

## Representation

Use a one-dimensional integer NumPy array `[a0, a1, ..., ad]` for
a0 + a1 X + ... + ad X^d. Every coefficient is a canonical label in `[0, q)`.
Trailing zeros are allowed and do not change the polynomial's value. Input
arrays must contain at least one coefficient. Results are integer arrays;
zero results can be represented as `[0]`, a padded zero vector, or
an empty integer array. Compare polynomial values independently of zero padding.

`add_poly`, `mult_poly`, and `divmod_poly` also accept Python integer labels as
constant polynomials in either operand. Input arrays must not be mutated.
Low-level operations assume valid coefficients. Malformed or empty inputs
have no validation guarantee.

## Operations

| Method | Contract |
| --- | --- |
| `add_poly(a, b)` | Coefficient-wise field addition; subtraction is identical in characteristic two. |
| `mult_poly(a, b)` | Polynomial convolution with field multiplication and addition. |
| `divmod_poly(a, b, verbose=False)` | Return `[quotient, remainder]` with a = b*quotient + remainder, and remainder zero or degree less than b. Must terminate for any nonzero divisor, including a constant. A zero divisor raises `ZeroDivisionError`. |
| `div_poly(a, b)`, `mod_poly(a, b)` | Return the quotient or remainder of the same division. |
| `is_factor(p, factorPoly)` | Return whether the remainder is zero; a zero factor raises `ZeroDivisionError`. |
| `derive_poly(p)` | Formal derivative: coefficient i contributes i*a_i at position i-1. In characteristic two this is a_i for odd i, zero for even i. Constants differentiate to zero. |
| `substitute_element_into_poly(p, a)` | Evaluate p at field label a and return the resulting integer label; evaluation at zero returns the constant coefficient. |
| `monic_multiplier(p)` | Return the inverse of the highest-degree nonzero coefficient, ignoring trailing zeros. Multiplying every coefficient by it makes the polynomial monic. A zero polynomial raises `ZeroDivisionError`. |

The formal derivative multiplies coefficients by integer degree; it does
**not** raise them to that degree.
