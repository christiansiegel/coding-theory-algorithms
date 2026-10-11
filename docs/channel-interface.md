# Binary symmetric channel calculations

This contract covers valid binary symmetric channels and the entropy, capacity,
and efficiency calculations tested in this step. Other discrete channels and
conditional-probability edge cases need further review.

## Inputs and conventions

Construct `BSC(PX, P)` with source probabilities `PX = [x, 1 - x]` and a
transition matrix `P = [[1 - p, p], [p, 1 - p]]`, where x and p are in [0, 1].
Rows of P correspond to input symbols; columns correspond to output symbols.
`P[i, j]` is P(Y=j | X=i). The constructor accepts lists or NumPy arrays and
stores floating-point arrays. Validation of malformed probabilities is not yet
part of the interface contract.

Entropy is measured in bits. Define h(z) = -z log2(z) - (1-z) log2(1-z), with
zero-probability terms contributing zero. Thus h(0) = h(1) = 0.

| Method | Contract for a binary symmetric channel |
| --- | --- |
| `PX()` | Return the source probabilities. |
| `PY()` | Return `[y, 1-y]`, where y = x(1-p) + (1-x)p. |
| `PYX()` | Return P transposed, with output symbols in rows. |
| `HX()` | Return h(x). |
| `HY()` | Return h(y). |
| `HYX()` | Return H(Y\|X) = h(p). |
| `Cs(verbose=True)` | Return capacity C = 1-h(p), using a uniform source without changing this instance's source probabilities. |
| `ChannelEfficiency(verbose=True)` | Return [h(y)-h(p)] / C. Raise `ZeroDivisionError` when C = 0 (p = 1/2). |

Passing `verbose=False` must suppress printed explanations. Entropy methods
default to quiet mode; capacity and efficiency retain their existing verbose
defaults. Deterministic sources and crossover probabilities 0 and 1 are valid.
At p = 1, bits are always flipped but perfectly recoverable, so capacity is 1.

## Limits and verification

The class also accepts larger or asymmetric transition matrices. Its `Cs()`
calculation evaluates mutual information at a uniform source, which need not
maximize mutual information for those matrices. General capacity optimization
is outside this step; results for such channels must not be assumed correct.

Tests use the closed-form binary entropy formulas independently of production
methods. They cover biased and deterministic sources, noiseless and fully
flipping channels, zero capacity, quiet/verbose output, instance independence,
and preservation of the source even when a capacity calculation raises.
