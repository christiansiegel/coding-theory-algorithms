# Binary linear and cyclic code interfaces

This contract covers small, nondegenerate binary codes with valid inputs.
BCH and Reed–Solomon behavior is not covered by this contract.

## Common conventions

Messages and codewords are one-dimensional integer NumPy arrays containing
bits. Their lengths are k and n respectively. Binary matrix arithmetic uses
modulo 2. Generator G has shape (k, n); parity check H has shape (n-k, n).
These interfaces expect the systematic arrangement `G = [P | I_k]` and
`H = [I_(n-k) | P.T]`. Thus information bits occupy the **last k positions**.
Conversions do not perform general Gaussian elimination on arbitrary matrices.

## LinearBlockCode

Create `LinearBlockCode()` then call `setG(G)` or `setH(H)` with a full-rank
systematic matrix. Using an uninitialized code, a rank-deficient matrix,
nonbinary data, or wrong input lengths has no established validation contract.

| Method | Contract |
| --- | --- |
| `G()`, `H()`, `P()` | Return the generator, parity check, or parity submatrix. |
| `k()`, `n()`, `R()` | Return dimensions and rate k/n. |
| `c(m)` | Return codeword mG modulo 2, with length n. |
| `s(r)` | Return syndrome rH.T modulo 2, with length n-k. |
| `M()`, `C()` | Enumerate all 2^k messages or codewords. Row i uses least-significant-bit-first representation of i for the message. |
| `dmin()` | Return the minimum weight of a nonzero codeword. |
| `t()` | Return floor((dmin-1)/2). |
| `errorDetectionCapability()` | Return dmin-1. |
| `Ai(i)` | Return the number of codewords of weight i, including the zero codeword for i=0. |
| `A()` | Preserve the README's existing n-entry layout: entry j counts weight j+1. It omits weight zero. |
| `correctableErrorPatterns()` | Enumerate every length-n error of weight at most t. |
| `syndromeDecode(r)` | Return a corrected length-n codeword for errors of weight at most t. Raise `KeyError` if the syndrome has no entry in the table. |
| `printErrorsThatHaveSyndrome(s)` | Print every length-n binary error vector e satisfying eH.T = s, with its Hamming weight. Return `None` and preserve s. Accept a list or array of n-k bits. |

The `A()` layout preserves the existing implementation and README example.

Beyond t errors, decoding may silently return a different codeword or raise
`KeyError`. A successful return is not a guarantee that the original message was
recovered. Enumeration is exponential and intended for small textbook examples.

The syndrome-printing helper enumerates all matching errors, including those
beyond the correction radius; it is a diagnostic, not a decoder. Zero syndrome
includes the zero vector and every nonzero codeword. Output spacing is not a
contract. Independent tests cover every syndrome in the repetition and Hamming
examples, using both list and array inputs. The former hardcoded `[0,0,1]`
selection is fixed; zero and nonzero requests now select their own error sets.

## CyclicCode

Construct `CyclicCode(g, n)` with a nonconstant binary generator polynomial
g that divides X^n+1, has constant coefficient 1, and degree r satisfying
0 < r < n. Polynomial vectors use ascending degree order. k = n-r.
Constructor checks are currently incomplete; callers must supply valid codes.

| Method | Contract |
| --- | --- |
| `g()` | Return the binary generator coefficient vector. |
| `c(m, systematic=True)` | Return the length-n coefficients of X^r m(X) + [X^r m(X) mod g(X)]. |
| `c(m, systematic=False)` | Return the length-n coefficients of m(X)g(X). |
| `G(systematic=True)` | Return a matrix giving the same encoding as `c()` with the corresponding flag. |
| `S(r)` | Return the polynomial remainder r(X) mod g(X). Trailing zero padding may vary; this is distinct from the matrix syndrome `s(r)`. |
| `shift(c, i=1)` | For nonnegative integer i, return length-n coefficients of X^i c(X) mod (X^n+1); shifting n positions returns the original vector. |
| `shiftSyndrome(S, i=1)` | For nonnegative integer i, return X^i S(X) mod g(X). |
| `dmin()`, `t()`, `errorDetectionCapability()` | Follow the same mathematical definitions as linear codes; generator weight is only an upper bound on minimum distance. |

Tests use fixed repetition/Hamming matrices and independent integer polynomial
arithmetic. They enumerate all messages and correctable errors in the selected
linear codes, and all received vectors for polynomial syndromes in the selected
cyclic codes. Wraparound shifts and the generator-weight counterexample have
separate regressions. Exact minimum distance enumerates 2^k messages; this can
be slow for larger codes. Probability methods, general systematic
conversion, invalid inputs, and BCH/RS still need independent coverage.
