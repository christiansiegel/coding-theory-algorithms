# Code quality work

Work proceeds one step at a time, with user review before each commit.

## Plan and status

1. **Baseline tooling and example tests — committed in `a0d7d4c`.** Add dependency
   configuration, a lockfile, import checks, and selected known-answer checks on
   the README script output. Keep algorithm files unchanged.
2. **Interface conventions and mathematical tests — finite-field element API
   committed in `b8c90d8`; linear/cyclic interfaces committed in `2ec4697`.** Document
   representations,
   shapes, systematic matrix layout, and invalid-input behavior. Test finite
   fields first, then linear/cyclic codes, BCH/RS, and channel calculations using
   independent answers, algebraic properties, and exhaustive small cases.
3. **Bug fixes — finite-field element fixes committed in `1990701`; NumPy
   compatibility fix committed in `6292409`; binary symmetric channel fixes
   committed in `272be01`; cyclic-code fixes awaiting review.**
4. **Formatting and linting.** Apply Ruff in a separate change.
5. **Package structure.** Move implementation into a package, preserving old
   imports with compatibility modules and keeping `playground.py` runnable.

## Baseline and validation

- Python requirement: 3.10 or newer; local validation currently covers only
  Python 3.13.
- NumPy requirement: `>=2.2,<3`. The compatibility fix was checked with 2.2.6
  and 2.5.4 on Python 3.13; the lockfile selects compatible versions for each
  Python version.
- Run `uv run python -m pytest` for the initial seven tests. They protect six
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
- [Binary code contract](docs/binary-code-interface.md), `tests/test_linear_codes.py`,
  and `tests/test_cyclic_codes.py` cover systematic matrices, enumeration,
  encoding, weight distributions, and exhaustive correction in small linear
  codes. Cyclic tests use independent binary-polynomial arithmetic for encoding,
  syndromes, distance, and shifts. The cyclic fixes remove all expected-failure
  markers and restore wraparound checks for all messages in the small examples.
  Exact distance enumeration is exponential in k and slower than the previous
  incorrect generator-weight shortcut. Current local result: **159 passed,
  no expected failures**, with NumPy 2.5.4. Ruff checks pass for the test files.
- The project runs from the repository root without being installed as a package.
- `uv run ruff check .` and `uv run ruff format --check .` expose existing source
  issues; passing them across the codebase is deferred to step 4.

## Confirmed issues to address

- For arbitrary transition matrices, `BSC.Cs()` assumes a uniform source is
  optimal. This is not always true: with `P = [[1, 0], [0.5, 0.5]]`, it returns
  approximately 0.311278 bits, while source `[0.6, 0.4]` achieves 0.321928 bits.
  General capacity optimization remains to be addressed.

## Fixes awaiting review

- Replace the cyclic generator-weight shortcut with exact linear
  minimum-distance enumeration using matrix encoding. For n=15 and
  g=1+X+X^2+X^3+X^4, distance is now
  2 rather than 5, and correction radius is 0 rather than 2. The weight-2
  codeword (1+X)g=1+X^5 proves the former shortcut wrong.
- Trim trailing zeros after cyclic shift reduction, before padding to length n.
- Correct `A()`'s docstring: entry j counts weight j+1 and weight zero is omitted.
  Its output layout remains unchanged.

## Completed fixes

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
