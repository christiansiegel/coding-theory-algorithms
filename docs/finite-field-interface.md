# Finite-field interface

This contract covers construction, element conversion, and element arithmetic
in `GaloisField.py`. Polynomial arithmetic and code interfaces will follow.
Known departures from the contract are tested with strict expected failures;
the implementation is unchanged in this step.

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

## Independent verification

Tests construct small fields GF(2), GF(4), GF(8), and GF(16). A reference uses
binary integer polynomials: XOR for addition, carryless multiplication, and
reduction by the defining polynomial. It does not call production arithmetic
or conversion methods to generate expected answers.

Every element conversion and every pair for addition and multiplication is
checked against that reference. Division by nonzero elements is checked for
all pairs except those requiring exponent -1; that known defect has a separate
regression test. Negative powers similarly isolate alpha^-1 from the passing
cases. Zero-handling is checked separately. This verifies the covered small fields,
not every possible primitive polynomial or the polynomial/code algorithms.
