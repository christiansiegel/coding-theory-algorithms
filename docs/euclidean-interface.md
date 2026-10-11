# Euclidean algorithm interfaces

The two methods named `HCF` have different purposes. These contracts retain
their existing names and calling conventions. Regression tests cover the
formerly failing edge cases without expected-failure markers.

## Integer greatest common divisor

`BCHCode.HCF(A, B)` is a module-level function accepting Python integers. It
returns the nonnegative greatest common divisor, including negative operands,
one zero operand, and `HCF(0, 0) == 0`. Computation must remain exact for
arbitrarily large integers; floating-point quotients are inappropriate.

The function prints its Euclidean calculation and the final result. Its existing
signature has no quiet option. Exact intermediate table spacing is not promised.
Tests use Python's integer `math.gcd` as an independent oracle.

## Polynomial decoder helper

`field.HCF(A, B, verbose=False)` is an early-stopping extended Euclidean
algorithm used by BCH/RS decoding, despite its historical GCD name. Inputs are
nonzero polynomial coefficient vectors in the supplied field's label
representation. Trailing zeros are allowed. Both input arrays are preserved.
Zero operands raise `ValueError`: the decoder handles zero syndromes separately.

The helper orders operands by degree, swapping them only if A has lower degree
than B. Let M denote the first operand after this ordering and S the second.
It returns `(r, t)`, with t the coefficient of **S after ordering**, satisfying
`r == t*S (mod M)`. It stops at the first Euclidean remainder for which r is
zero or `degree(r) < degree(t)`. A zero remainder must terminate even though
the shared `degree()` utility historically reports degree zero for `[0]`.

The result need not be monic and need not be the polynomial GCD. Callers may
normalize it afterward, as the decoder does. Returned trailing zero padding
may vary. `verbose=False` prints nothing; verbose mode prints the calculation.

Tests check independent known answers, the modular identity using binary
polynomial arithmetic, stopping degrees, operand ordering, input preservation,
and zero/constant/equal-operand cases. Decoder regressions remain in the BCH/RS
suite to protect how this helper is used.
