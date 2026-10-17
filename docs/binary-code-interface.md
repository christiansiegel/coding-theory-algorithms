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

The common textbook arrangement `G = [I_k | P]`, `H = [P.T | I_(n-k)]`
is equally valid over GF(2). Both arrangements satisfy `G @ H.T = 0` modulo 2:
for this library's arrangement the product is `P + P = 0`. The vertical bar
denotes horizontal concatenation of column blocks.

Moving the information block from the front to the back in **both** matrices
converts the textbook arrangement to this library's arrangement. Apply the same
column permutation to codewords, received vectors, and error vectors. The result
is an equivalent code with reordered coordinates and the same distances.

`GtoH`, `HtoG`, `GtoP`, and `HtoP` in
`coding_theory_algorithms.linear_block_code` assume the library's layout; they
do not detect or convert the alternative layout automatically. `GtoP` takes the
left k-by-(n-k) block, while `HtoP` transposes the right (n-k)-by-k block.

## LinearBlockCode

Create `LinearBlockCode()` then call `set_g(G)` or `set_h(H)` with a full-rank
systematic matrix. Setters and code-vector boundaries validate dimensions,
systematic layout, and coefficients; see [input validation](input-validation.md).
Initialize the code before using its arithmetic methods.

| Method | Contract |
| --- | --- |
| `G()`, `H()`, `P()` | Return the generator, parity check, or parity submatrix. |
| `k()`, `n()`, `R()` | Return dimensions and rate k/n. |
| `c(m)` | Return codeword mG modulo 2, with length n. |
| `s(r)` | Return syndrome rH.T modulo 2, with length n-k. |
| `M()`, `C()` | Enumerate all 2^k messages or codewords. Row i uses least-significant-bit-first representation of i for the message. |
| `dmin()` | Return the minimum weight of a nonzero codeword. |
| `t()` | Return floor((dmin-1)/2). |
| `error_detection_capability()` | Return dmin-1. |
| `Ai(i)` | Return the number of codewords of weight i, including the zero codeword for i=0. |
| `A()` | Return n entries: entry j counts weight j+1. It omits weight zero. |
| `PU(p)` | Return the probability of a nonzero error with zero syndrome under independent bit flips of probability p. |
| `Pe(p)` | Return the probability of more than t bit errors under independent bit flips of probability p. |
| `correctable_error_patterns()` | Enumerate every length-n error of weight at most t. |
| `syndrome_decode(r)` | Return a corrected length-n codeword for errors of weight at most t. Raise `KeyError` if the syndrome has no entry in the table. |
| `print_errors_that_have_syndrome(s)` | Print every length-n binary error vector e satisfying eH.T = s, with its Hamming weight. Return `None` and preserve s. Accept a list or array of n-k bits. |
| `print_standard_array()` | Print all 2^(n-k) cosets with 2^k vectors per row, covering every length-n vector exactly once. The first column contains minimum-weight coset leaders; the first row is `C()` in message order. Return `None` and preserve G. |

For probability methods, p is a real number in [0, 1], including both endpoints.
`PU(p)` excludes the no-error event: it sums `p**w * (1-p)**(n-w)` over nonzero
codewords of weight w. `Pe(p)` is the binomial tail above the guaranteed
correction radius t. It measures the probability of exceeding that guarantee;
it does not specify the failure rate of an arbitrary decoder beyond t.
Both methods are quiet and preserve the code matrices. Invalid probabilities
raise `ValueError`.

Beyond t errors, decoding may silently return a different codeword or raise
`KeyError`. A successful return is not a guarantee that the original message was
recovered. Enumeration is exponential and intended for small textbook examples.

The standard array includes cosets whose leaders exceed the correction radius.
It is a complete diagnostic; `correctable_error_patterns()` and `decoding_table()`
remain restricted to errors of weight at most t. Each row is its leader plus
every codeword modulo 2, in the same column order as the first row. Ties between
minimum-weight leaders, coset row order, and output spacing are not contracts.

The syndrome-printing helper enumerates all matching errors, including those
beyond the correction radius; it is a diagnostic, not a decoder. Zero syndrome
includes the zero vector and every nonzero codeword. Output spacing is not a
contract.

## CyclicCode

Construct `CyclicCode(g, n)` with a nonconstant binary generator polynomial
g that divides X^n+1, has constant coefficient 1, and degree r satisfying
0 < r < n. Polynomial vectors use ascending degree order. k = n-r.
Invalid generators and dimensions raise `ValueError`.

| Method | Contract |
| --- | --- |
| `g()` | Return the binary generator coefficient vector. |
| `c(m, systematic=True)` | Return the length-n coefficients of X^r m(X) + [X^r m(X) mod g(X)]. |
| `c(m, systematic=False)` | Return the length-n coefficients of m(X)g(X). |
| `G(systematic=True)` | Return a matrix giving the same encoding as `c()` with the corresponding flag. |
| `S(r)` | Return the polynomial remainder r(X) mod g(X). Trailing zero padding may vary; this is distinct from the matrix syndrome `s(r)`. |
| `shift(c, i=1)` | For nonnegative integer i, return length-n coefficients of X^i c(X) mod (X^n+1); shifting n positions returns the original vector. |
| `shift_syndrome(S, i=1)` | For nonnegative integer i, return X^i S(X) mod g(X). |
| `dmin()`, `t()`, `error_detection_capability()` | Follow the same mathematical definitions as linear codes; generator weight is only an upper bound on minimum distance. |

Exact minimum distance enumerates 2^k messages and can be slow for larger
codes. BCH/RS have a [separate contract](bch-rs-interface.md).
