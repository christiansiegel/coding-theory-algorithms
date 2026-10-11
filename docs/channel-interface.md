# Discrete channel calculations

This contract covers binary symmetric channels and extends capacity and
efficiency to valid finite discrete channels. Conditional-probability edge
cases need further review.

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

## General channel capacity

The class also accepts larger or asymmetric transition matrices. For these,
`PX` has one entry per input row, all probabilities are nonnegative and finite,
and each input row sums to one. The source also sums to one. Rectangular
matrices, duplicate input rows, and output columns that are always zero are
valid. Malformed-input validation remains outside this contract.

`Cs()` must maximize I(X;Y) over all input probability distributions, in bits,
without changing this instance's source. A uniform source is optimal for a
binary symmetric channel, but need not be optimal in general.
`ChannelEfficiency()` returns this source's mutual information divided by the
optimized capacity, and raises `ZeroDivisionError` at zero capacity. Quiet and
verbose behavior follows the same conventions as above.

For the asymmetric Z channel `P = [[1,0],[p,1-p]]`, with 0 < p < 1, capacity is
`log2(1 + (1-p)*p**(p/(1-p)))`. At p=1/2 it is `log2(1.25)` (about 0.321928
bits), achieved by source `[0.6,0.4]`. Evaluating the uniform source gives only
about 0.311278 bits; using that as capacity incorrectly reports efficiency
above 1 at the optimal source. The optimizer fixes this former defect.

Capacity uses Blahut–Arimoto iterations with lower and upper capacity bounds.
It returns the lower bound when their gap is at most 1e-12 bits. Calculations
use log probabilities to retain rare inputs and omit unreachable outputs.
Identical input rows give capacity zero. If 100,000 iterations do not establish
the tolerance, `Cs()` raises `RuntimeError`; efficiency propagates that error.
The source is preserved on both success and failure. No new dependency is used.
For boundary optima, the optimizer also tries removing currently unhelpful
inputs. Such a candidate is accepted only when the bounds over **all** input
rows certify the same tolerance; unsuccessful candidates do not alter the
ongoing iteration.

## Verification

Tests use the closed-form binary entropy formulas independently of production
methods. They cover biased and deterministic sources, noiseless and fully
flipping channels, zero capacity, quiet/verbose output, instance independence,
and preservation of the source even when a capacity calculation raises.
General-channel tests use the Z-channel closed form and independently known
capacities for distinguishable deterministic outputs and identical input rows.
They include rectangular matrices, unreachable outputs, unhelpful input rows,
and permutations of input/output symbols. All capacity and efficiency
regressions now pass. A reduced iteration limit checks that nonconvergence
raises without changing the stored source.
The channel file's own three-input/five-output example checks a boundary
optimum with capacity exactly 2/3 bits and optimal source `[1/2,0,1/2]`.
