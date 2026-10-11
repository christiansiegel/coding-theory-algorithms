# Code quality work

Work proceeds one step at a time, with user review before each commit.

## Plan and status

1. **Baseline tooling and example tests — committed in `a0d7d4c`.** Add dependency
   configuration, a lockfile, import checks, and selected known-answer checks on
   the README script output. Keep algorithm files unchanged.
2. **Interface conventions and mathematical tests — finite-field element API
   committed in `b8c90d8`; linear/cyclic interfaces committed in `2ec4697`;
   polynomial arithmetic committed in `f6be7a6`; BCH/RS contracts and tests
   committed in `7b50bbc`.** Document
   representations,
   shapes, systematic matrix layout, and invalid-input behavior. Test finite
   fields first, then linear/cyclic codes, BCH/RS, and channel calculations using
   independent answers, algebraic properties, and exhaustive small cases.
3. **Bug fixes — finite-field element fixes committed in `1990701`; NumPy
   compatibility fix committed in `6292409`; binary symmetric channel fixes
   committed in `272be01`; cyclic-code fixes committed in `7e8bdc5`;
   polynomial fixes committed in `8eb6b7b`; BCH construction and shared
   decoding fixes committed in `507b412`; RS encoding and distance fixes
   committed in `cb124be`.** General channel-capacity contract and regression
   tests await review.
4. **Formatting and linting.** Apply Ruff in a separate change.
5. **Package structure.** Move implementation into a package, preserving old
   imports with compatibility modules and keeping `playground.py` runnable.

## Baseline and validation

- Python requirement: 3.10 or newer; local validation currently covers only
  Python 3.13.
- NumPy requirement: `>=2.2,<3`. The compatibility fix was checked with 2.2.6
  and 2.5.4 on Python 3.13; the lockfile selects compatible versions for each
  Python version.
- Run `uv run python -m pytest` for the full suite. The initial seven tests protect six
  existing imports and selected numerical results in the two printed examples;
  they do not establish correctness across all algorithms or inputs.
- [Finite-field contract](docs/finite-field-interface.md) and
  `tests/test_galois_field.py` cover element arithmetic in GF(2), GF(4), GF(8),
  and GF(16) using an independent binary-polynomial reference. The fixes remove
  all expected-failure markers and restore exhaustive division and negative
  power checks.
- `tests/test_binary_vectors.py` checks bit order, zero-padding, all 256
  eight-bit values, and an integer larger than 64 bits. Known-answer tests
  reject deprecated NumPy calls. The 72-test suite at that step passed with
  both NumPy versions above.
  An existing `is 0` SyntaxWarning appeared when recompiling `LinearBlockCode.py`;
  that separate source issue remains.
- [Channel contract](docs/channel-interface.md) and
  `tests/test_binary_symmetric_channel.py` check closed-form entropy, capacity,
  and efficiency results for binary symmetric channels, including boundary
  probabilities, verbosity, and source preservation.
  General-capacity tests add independent Z-channel answers, duplicate input
  rows, rectangular matrices, and unreachable outputs. Five strict expected
  failures demonstrate underestimated capacity and efficiency above one.
  Current full-suite result: **273 passed, 5 expected failures**, with NumPy 2.5.4.
- [Binary code contract](docs/binary-code-interface.md), `tests/test_linear_codes.py`,
  and `tests/test_cyclic_codes.py` cover systematic matrices, enumeration,
  encoding, weight distributions, and exhaustive correction in small linear
  codes. Cyclic tests use independent binary-polynomial arithmetic for encoding,
  syndromes, distance, and shifts. The cyclic fixes remove all expected-failure
  markers and restore wraparound checks for all messages in the small examples.
  Exact distance enumeration is exponential in k and slower than the previous
  incorrect generator-weight shortcut.
- [Polynomial contract](docs/polynomial-interface.md) and the polynomial tests
  in `tests/test_galois_field.py` cover addition, multiplication, division,
  evaluation, formal derivatives, and monic multipliers using independent field
  arithmetic. The fixes remove all expected-failure markers; constant-divisor
  regressions remain isolated in subprocesses with timeouts and now cover scalar
  and array divisors in all four fields.
- [BCH/RS contract](docs/bch-rs-interface.md) and `tests/test_bch_rs_codes.py`
  cover generators, independent encodings/syndromes, and small decoding cases.
  The arithmetic oracle was moved unchanged into `tests/field_reference.py`
  for reuse. BCH construction and shared decoding regressions now pass,
  including all single-symbol errors in RS(3,1). Expanded parity checks cover
  BCH(7,4) and BCH(15,7), including generator/check-matrix orthogonality.
  RS encoding tests exhaust all messages in RS(3,1) over GF(4) and RS(7,3)
  over GF(8), in both encoding forms, and independently enumerate their
  minimum symbol weights. GF(8) syndrome and decoding checks include nonbinary
  symbols and selected one- and two-error patterns. All expected-failure
  markers have been removed from BCH/RS tests. Ruff checks pass for the test files.
- The project runs from the repository root without being installed as a package.
- `uv run ruff check .` and `uv run ruff format --check .` expose existing source
  issues; passing them across the codebase is deferred to step 4.

## Tests awaiting review

- Extend the channel contract to define capacity as maximum mutual information
  over source distributions for valid finite discrete transition matrices.
- Add independent known-answer tests before changing the capacity calculation.
  Symmetric and deterministic boundary cases pass; asymmetric capacity,
  optimal-source efficiency, and duplicate-row regressions fail as expected.

## Confirmed issues to address

- For arbitrary transition matrices, `BSC.Cs()` assumes a uniform source is
  optimal. This is not always true: with `P = [[1, 0], [0.5, 0.5]]`, it returns
  approximately 0.311278 bits, while source `[0.6, 0.4]` achieves 0.321928 bits.
  General capacity optimization remains to be addressed.

## Completed fixes

- Encode RS messages with the supplied field's polynomial arithmetic.
- Return the primitive RS minimum symbol distance `2t+1` and design radius t,
  avoiding binary enumeration and correcting the syndrome count.

- Return a copy of an error-free BCH/RS codeword when all syndromes are zero,
  avoiding division by zero in Euclid.
- Allocate the decoder's error vector by code length, so an error that cancels
  the highest nonzero received coefficient remains correctable.
- Expand field labels into polynomial-basis bits in `BCHCode.H(False)`.
- Include each conjugacy class once in BCH generator construction. For GF(16),
  design t=5 now produces the degree-14 repetition-code generator.

- Stop polynomial division when the residual is zero, fixing constant divisors.
  Initialize the quotient as an integer zero polynomial.
- Convert scalar second operands into the second array in addition,
  multiplication, and division.
- Raise `ZeroDivisionError` for zero polynomial divisors.
- Compute formal derivatives by retaining every odd-degree coefficient,
  including the highest term, and return an integer array.
- Invert the highest-degree nonzero coefficient for monic multipliers,
  ignoring trailing zeros. The zero polynomial now raises `ZeroDivisionError`.

- Replace the cyclic generator-weight shortcut with exact linear
  minimum-distance enumeration using matrix encoding. For n=15 and
  g=1+X+X^2+X^3+X^4, distance is now
  2 rather than 5, and correction radius is 0 rather than 2. The weight-2
  codeword (1+X)g=1+X^5 proves the former shortcut wrong.
- Trim trailing zeros after cyclic shift reduction, before padding to length n.
- Correct `A()`'s docstring: entry j counts weight j+1 and weight zero is omitted.
  Its output layout remains unchanged.

- Compute `ChannelEfficiency()` from this instance rather than the module's
  `bsc` example variable. Raise `ZeroDivisionError` at zero capacity.
- Treat zero-probability terms as zero in `HX()` and `HY()`, avoiding NaN.
- Honor `verbose=False` in `Cs()` and compute its uniform-source result using
  a separate channel instance, preserving the original source even on failure.

- Replace the removed binary `np.fromstring` mode in `intToArray()` with
  `np.frombuffer` on ASCII bytes. Preserve least-significant-bit-first order,
  zero-padding, and uint8 output. Widen the NumPy constraint and update the
  lockfile after testing both the original baseline and newer NumPy.

- Normalize finite exponents modulo q - 1 before converting to labels. This
  fixes `elementFromExp(-1)`, division with exponent difference -1, and alpha^-1.
- Raise `ZeroDivisionError` for division by zero, including 0/0.
- Return 1 for `powElement(0, 0)`, consistent with the reviewed contract and
  polynomial evaluation; raise `ZeroDivisionError` for negative powers of zero.

The README also spells GF(2^5) as GF(25) in its first exercise; correct that when
reviewing the mathematical documentation.
