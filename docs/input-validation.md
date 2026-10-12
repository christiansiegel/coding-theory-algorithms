# Public input validation

Construction and encoding boundaries raise `ValueError` for malformed inputs.
Array inputs are copied; finite integral numeric coefficients (including 0.0
and 1.0) are accepted, then stored as integers.

- Channels require a nonempty source and matching nonempty transition matrix.
  Values must be real numeric, finite, and nonnegative; the source and each row sum to one
  within absolute tolerance 1e-12. Inputs are not silently normalized.
- Fields require a positive-degree primitive binary polynomial. Trailing zeros
  are trimmed. Nonprimitive and reducible polynomials are rejected.
- Linear-code setters require binary systematic matrices, with 0 < rows < n:
  G uses [P | I], and H uses [I | P.T]. Invalid setters preserve the old code.
- Cyclic generators must be nonconstant binary divisors of X^n+1, with constant
  coefficient one and degree less than integer n. Cyclic matrix setters raise
  `NotImplementedError` because the constructor defines the generator.
- BCH/RS require a field instance, integer t >= 1, 2t < n=q-1, and positive
  message dimension.
- Linear messages have exactly k bits; linear received vectors have n bits.
  Cyclic/BCH and RS polynomial messages may have 1 through k coefficients.
  BCH/RS received vectors have n symbols. RS labels must lie in [0,q).
- PU/Pe accept only finite scalar probabilities in [0,1].

These checks cover public construction, matrix setters, and encoding/decoding
boundaries. Low-level polynomial arithmetic retains the coefficient conventions
and preconditions in its separate contract; arbitrary malformed low-level
operands do not have a broader validation guarantee.
