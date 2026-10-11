# BCH and Reed–Solomon interfaces

These contracts describe intended behavior for small primitive narrow-sense
codes. Strict expected-failure tests record current implementation defects.
Invalid parameters, shortened codes, and decoding beyond the guaranteed
correction radius remain outside this step.

## Binary BCH codes

`BCHCode(field, t, verbose=False)` constructs a length n = 2^m-1 binary code
using the primitive element of the supplied GF(2^m). The designed distance is
2t+1. Its generator is the least common multiple of the binary minimal
polynomials for alpha^1 through alpha^(2t). Each distinct factor occurs once,
even when multiple requested roots belong to the same conjugacy class.

Messages and codewords are binary coefficient vectors in ascending degree
order. Encoding, systematic layout, and exact minimum distance follow the
[binary-code contract](binary-code-interface.md). The actual correction radius
`t()` is floor((dmin-1)/2), which can exceed the constructor's design parameter.

| Method | Contract |
| --- | --- |
| `GF()` | Return the field used for roots and syndrome arithmetic. |
| `g()`, `n()`, `k()` | Return the binary generator, length n, and dimension n-degree(g). |
| `H()` | Return the systematic binary parity check matrix. |
| `H(False)` | Return a binary-expanded check matrix. For each odd exponent 1,3,...,2*t()-1, column j contains the m polynomial-basis coefficients of alpha^(exponent*j). Stack those blocks vertically. |
| `S(r, verbose=True)` | Return 2*t() field labels for r(alpha^1), ..., r(alpha^(2*t())). Unlike the cyclic `S()` method, this is a vector of evaluations, not a binary polynomial remainder. |
| `decode(r, verbose=True)` | Return the corrected codeword polynomial for up to t() errors, including zero errors. The input is a length-n vector; returned trailing zero padding may vary. |

`verbose=False` must suppress printed explanations. Returned codewords are
compared as polynomial values, allowing trailing zeros to be omitted. An error
that cancels the highest nonzero coefficient is still correctable; the decoder
must retain enough storage to address every code position.

## Reed–Solomon codes

`RSCode(field, t, verbose=False)` constructs a length n = q-1 code **over the
supplied field**, with k = n-2t and minimum symbol distance 2t+1. Its generator
is the product of (X+alpha^i) for i=1,...,2t. Coefficients, messages, received
vectors, and codewords contain field-element labels in `[0,q)`, not just bits.
See the [field representation contract](finite-field-interface.md).

| Method | Contract |
| --- | --- |
| `g()`, `n()`, `k()` | Return the field-valued generator and symbol dimensions. |
| `c(m, systematic=True)` | Encode using the same polynomial formulas as cyclic codes, with arithmetic in the supplied field. Output has n symbols. |
| `c(m, systematic=False)` | Return the length-n coefficients of m(X)g(X) over the supplied field. |
| `dmin()`, `t()` | Return 2t+1 and t for valid nondegenerate primitive RS parameters. Distance counts nonzero symbols, and any enumeration must cover field-valued messages. |
| `S(r, verbose=True)` | Evaluate r at alpha^1,...,alpha^(2t), returning 2t field labels. |
| `decode(r, verbose=True)` | Correct up to t erroneous symbols, including zero errors and arbitrary nonzero error values. Returned zero padding may vary. |

No binary-only matrix or probability methods are promised for RS in this step.
The implementation currently inherits several such methods; they need separate
review before use with field-valued symbols.

## Verification

Tests share the independent field-polynomial reference with the arithmetic
tests. BCH generators are constructed from the union of Frobenius conjugate
roots, and RS generators from distinct consecutive roots. Expected encodings
use independent polynomial division; expected syndromes use Horner evaluation.

Passing cases cover BCH(7,4) and BCH(15,7) encodings, every binary received
vector's BCH(7,4) syndromes, selected single-error BCH decodings and two-error
BCH(15,7) patterns, RS generators
over GF(4)/GF(8), every GF(4) received vector's RS(3,1) syndromes, and its
single-symbol-error corrections except the isolated cancellation failure.
Strict expected failures cover error-free decoding, canceled highest terms,
BCH factor deduplication and expanded parity checks, and RS encoding/distance.
