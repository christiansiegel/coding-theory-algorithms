# Finite-field interface

This contract covers construction, element conversion, and element arithmetic
in `GaloisField.py`. See the separate [polynomial contract](polynomial-interface.md)
for polynomial arithmetic.
Regression tests cover negative exponents, division by zero, and powers of zero.

## Construction and representations

`GaloisField(p)` constructs GF(2^m), where `p` is a one-dimensional integer
NumPy array giving a primitive binary polynomial of degree m >= 1.
Coefficients are ordered from lowest to highest degree: `[1, 1, 0, 0, 1]`
means 1 + X + X^4. Callers currently need to provide valid primitive polynomials;
validation of malformed or nonprimitive polynomials is not guaranteed.
`GF2` is the predefined GF(2) instance.

An element's integer label is **not its binary polynomial value**:

- Label 0 is the zero element.
- Label 1 is the multiplicative identity.
- Label a, for 1 <= a < q, represents alpha^(a - 1), with q = 2^m.
- Alpha is the root of the defining polynomial. Nonzero exponents are periodic
  modulo q - 1. In GF(2), the only nonzero element is 1.

For example, in GF(16) defined by 1 + X + X^4, label 5 represents alpha^4 =
1 + alpha. Its polynomial vector is `[1, 1, 0, 0]`, whose binary value is 3.

## Element API

| Method | Contract |
| --- | --- |
| `m()`, `q()` | Return m and 2^m. |
| `p()` | Return the defining polynomial as an integer array. |
| `element(a)` | Keep label 0 as zero; normalize any other integer label to `1 + ((a - 1) % (q - 1))`. This is label normalization, not exponent conversion. |
| `elementToExp(a)` | Return the normalized label's exponent; zero maps to negative infinity. |
| `elementFromExp(e)` | Return the label of alpha^e for any integer e, including negative integers. Either infinity maps to zero for compatibility with the existing API. |
| `elementToPoly(a, size_m=False)` | Return the normalized element as an integer coefficient vector in ascending order. Current output has m coefficients, including zero padding. `size_m=True` also guarantees that width. |
| `elementFromPoly(v)` | Return the label of a binary coefficient vector of degree less than m. Trailing zeros are allowed. This method does not promise reduction of higher-degree polynomials. |
| `addElements(a, b)` | Return the integer label of a + b in the field. |
| `multElements(a, b)` | Return the integer label of a * b in the field. |
| `divElements(a, b)` | Return the integer label of a / b for nonzero b; raise `ZeroDivisionError` when b is zero. |
| `powElement(a, e)` | Return the integer label of a^e for integer e. Nonzero a allows negative e. Define a^0 = 1, including 0^0; zero to a negative power raises `ZeroDivisionError`. |

Arithmetic tests use canonical labels in `[0, q)`. Inputs such as floats,
out-of-range polynomial coefficients, and empty vectors have no established
validation contract yet. These functions should not mutate caller-owned inputs.

## Root helpers

These methods accept canonical element labels in [0, q). Polynomial
coefficients are field labels in ascending degree order, as in the polynomial
contract. Caller-owned polynomials and root lists must be preserved.

| Method | Contract |
| --- | --- |
| `roots(p)` | Return a sorted list of distinct nonzero field roots. Zero is excluded even if p(0)=0; the zero polynomial returns all nonzero elements. |
| `conjugateRoots(root, verbose=False)` | Return the sorted, distinct Frobenius orbit under repeated squaring, including the supplied root. Zero and one each have singleton orbits. Quiet/verbose calls return the same group. |
| `conjugateRootGroups()` | Return every distinct orbit once, partitioning all q elements, including zero. Group order is not an API contract. |
| `removeConjugateRoots(roots)` | Return a new sorted list containing the lowest supplied label in each represented orbit. Accept an unsorted list with duplicates; preserve it. A singleton remains unchanged in value even if its orbit contains a lower absent label. |
| `minimalPolynomial(roots)` | Given a complete, distinct Frobenius orbit as a list or array, return the monic product of (X+root) as an integer array. Its coefficients are binary. For [0], return X; for [1], return X+1. |

`roots()` retains the nonzero search used by the decoder's Chien search.
`minimalPolynomial()` expects one complete orbit; arbitrary or repeated roots
have no established minimal-polynomial contract.

## Binary polynomial classification

`irreducible(p, verbose=False)` and `primitive(p, verbose=False)` classify a
nonempty one-dimensional integer array of binary coefficients over GF(2).
They use the degree of p, ignoring trailing zeros, and return Python booleans.
Their answers do not depend on the receiver's defining polynomial or degree.
These are binary-polynomial checks, even when called on an extension field.

Zero and constant polynomials are neither irreducible nor primitive. A
positive-degree polynomial is irreducible if it has no nonconstant proper
factor over GF(2). It is primitive if it is irreducible and X modulo p has
multiplicative order `2**degree(p)-1`. Thus X is irreducible but not primitive,
and X+1 is both. Quiet calls print nothing; verbose calls explain the same
boolean result. Inputs are preserved. Nonbinary or malformed inputs have no
established validation contract.

## Independent verification

Tests construct small fields GF(2), GF(4), GF(8), and GF(16). A reference uses
binary integer polynomials: XOR for addition, carryless multiplication, and
reduction by the defining polynomial. It does not call production arithmetic
or conversion methods to generate expected answers.

Every element conversion and every pair for addition, multiplication, and
division by nonzero elements is checked against that reference. Negative powers
and zero-handling are checked separately. This verifies the covered small fields,
not every possible primitive polynomial or the polynomial/code algorithms.
Root tests check every monic linear and quadratic polynomial in these fields,
using independent evaluation. Orbit and minimal-polynomial tests use reference
squaring and polynomial products. All root-helper regressions now pass,
including complete GF(2) coverage and conjugate reduction that preserves inputs
and supplied representatives.

Binary classification tests enumerate all 64 polynomials of degree at most
five, using independent bit-mask factorization and multiplicative-order checks.
They exercise every receiver field, trailing-zero padding, and quiet/verbose
calls. Strict expected failures record the degree-dependence and constant cases.
