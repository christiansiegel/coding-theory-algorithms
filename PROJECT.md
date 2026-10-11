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
   committed in `cb124be`.** General channel-capacity contract and tests
   committed in `16b0188`; capacity optimization committed in `8f128d6`.
4. **Formatting and linting — Ruff formatting committed in `6e7ac33`.**
   Routine lint cleanup committed in `9fe7c07`. Euclidean contracts and
   independent edge-case tests committed in `e891a53`; Euclidean fixes committed
   in `e33002a`.
5. **Package structure — complete.** Move implementations into a source
   package, preserving old imports with compatibility modules and keeping
   `playground.py` runnable. Dependency installation remains unchanged.
6. **Local checks — complete.** Add one command for lint,
   formatting checks, and tests, plus individual targets. Keep checks local.
7. **Untested-method audit — syndrome-printing contracts and tests committed in
   `0cddc10`; fix committed in `f749436`. Conditional-probability contracts and
   tests awaiting review.**
   Extend contracts and independent tests for uncovered public methods, then
   fix confirmed defects in separate reviews.

## Conditional-probability contracts and tests awaiting review

- Specify posterior matrix orientation and Bayes' rule for reachable outputs.
  Use `NaN` columns to mark undefined conditioning on unreachable outputs,
  without numerical warnings; exclude those outputs from conditional entropy.
- Test explicit posterior matrices for biased, uninformative, deterministic,
  noiseless, and rectangular channels. Compute conditional entropy independently
  as joint entropy minus output entropy and check the entropy chain rule.
- Check quiet/verbose behavior and preservation of stored probabilities.
- Six strict expected failures record division errors in `PXY()` and `HXY()`
  for three unreachable-output cases. Production code is unchanged.
- Validation: `make check` passes lint and formatting checks, with **331 tests
  passed and 6 strict expected failures** recording the confirmed defect.

## Completed syndrome-printing fix

- Define `printErrorsThatHaveSyndrome(s)` as an exhaustive diagnostic: print all
  vectors whose syndrome equals s, including errors beyond the correction
  radius. Preserve list/array inputs and report each vector's actual weight.
- Use fixed repetition/Hamming check matrices and independent integer parity
  to check the complete output sets for every syndrome, without snapshotting
  whitespace or imposing output ordering.
- Keep the requested syndrome separate from each candidate's computed syndrome
  and compare them, replacing the hardcoded `[0,0,1]` selection.
- Enable the three regression cases covering all syndromes for repetition and
  Hamming codes. The existing `[0,0,1]` results remain covered.
- Malformed syndrome validation remains outside this step, consistent with the
  existing valid-input code contract.
- Validation: `make check` passes lint and formatting checks, with **321 tests
  passed** and no expected failures.

## Completed local checks

- `make` and `make check` run Ruff lint, formatting checks, and pytest.
- Individual targets are `test`, `lint`, `format-check`, and `format`.
  Only `format` applies formatting changes; the check targets preserve source.
- Commands use `uv run --locked` to preserve the dependency lockfile and
  propagate errors through Make. `UV` can be overridden when needed.
- Validation: `make check` passes lint, formatting checks, and **317 tests**.
  Each target propagates a simulated command failure, and check stops on a
  failed lint step. Source, configuration, and lockfile contents are unchanged.

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
  The existing `is 0` SyntaxWarning is resolved by the routine lint cleanup.
- [Channel contract](docs/channel-interface.md) and
  `tests/test_binary_symmetric_channel.py` check closed-form entropy, capacity,
  and efficiency results for binary symmetric channels, including boundary
  probabilities, verbosity, and source preservation.
  General-capacity tests add independent Z-channel answers, duplicate input
  rows, rectangular matrices, and unreachable outputs. The optimizer removes
  all five expected failures. Additional checks cover unhelpful inputs, symbol
  permutations, verbose output, and source preservation on nonconvergence.
  The channel script's boundary optimum is checked independently at 2/3 bits.
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
- [Project layout](docs/project-layout.md) maps the six implementation modules
  into `coding_theory_algorithms/`. Root compatibility modules share classes,
  singleton fields, and module attributes. Fresh-process tests cover both
  import orders and script entry points.
- [Euclidean contracts](docs/euclidean-interface.md) distinguish integer GCD
  from the early-stopping polynomial decoder helper. Tests compare integers
  against `math.gcd` and verify polynomial known answers and modular identities
  using independent arithmetic. Fixes remove all expected-failure markers for
  zero/sign/large-integer and polynomial zero-remainder defects, and implement
  the reviewed validation of zero polynomial operands. Signed integer checks
  now cover every pair from -25 to 25; verbose output covers the zero stop.
- `uv run ruff format --check .` and `uv run ruff check .` now pass across
  the codebase with the 200-character line-length target. Intentional unused
  playground imports retain their file-level exemption.

## Completed package structure

- Move six implementations into snake-case package modules and convert their
  internal dependencies to relative imports. AST comparison confirms that
  algorithm statements are unchanged beyond those import paths.
- Keep the root filenames as compatibility entry points sharing their package
  modules; preserve script execution. Keep playground imports intact.
- Add fresh-process compatibility tests for class identity, module state,
  import ordering, GF2 singleton identity, and known channel script results.
- Add a short README import example and a detailed layout/compatibility document.
  The uv configuration and lockfile are unchanged.
- Validation: **317 tests passed with no warnings or expected failures**;
  repository-wide Ruff lint and formatter checks pass. All six moved
  implementation ASTs match the committed code apart from relative imports.

## Completed fixes

- Initialize Euclidean state explicitly in both functions, resolving the
  remaining undefined/unused-variable diagnostics.
- Normalize integer signs, handle `(0,0)`, and use exact integer quotients.
  Precision and overflow regressions now pass.
- Stop polynomial Euclid on a zero remainder, including equal and constant
  operands. Reject zero operands with `ValueError`. Preserve early stopping,
  degree ordering, modular identity, and the decoder's use of the helper.
- Remove all 11 expected-failure markers, broaden signed integer checks, and
  check verbose zero-remainder output. Full lint and formatter checks pass.
- Validation: **302 tests passed with no warnings or expected failures**,
  including all BCH/RS decoder regressions, with NumPy 2.5.4.

- Remove unused imports/root-list construction, rename the conjugation counter,
  and use numeric equality in standard-array printing. Preserve method calls
  whose assignment targets were unused and retain intentional playground imports.
- Routine lint validation: 282 tests passed without warnings; formatting passes.

- Optimize finite discrete channel capacity with Blahut–Arimoto iterations,
  using log probabilities and omitting unreachable outputs. Preserve the
  instance's source. Identical input rows return exactly zero capacity.
- Stop when the upper/lower capacity bounds differ by at most 1e-12 bits;
  raise `RuntimeError` after 100,000 iterations if that bound is not reached.
- Try reduced input supports to accelerate boundary optima, accepting them
  only when capacity bounds over all input rows certify the same tolerance.
- Replace the previous uniform-source result for asymmetric channels. For
  `P = [[1, 0], [0.5, 0.5]]`, capacity is now approximately 0.321928 bits,
  and source `[0.6, 0.4]` has efficiency one.

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
