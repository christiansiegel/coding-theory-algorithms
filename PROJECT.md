# Code quality work

Work proceeds one step at a time, with user review before each commit.

## Plan and status

1. **Baseline tooling and example tests — committed in `a0d7d4c`.** Add dependency
   configuration, a lockfile, import checks, and selected known-answer checks on
   the README script output. Keep algorithm files unchanged.
2. **Interface conventions and mathematical tests — finite-field element API
   committed in `b8c90d8`; other interfaces remain to be covered.** Document
   representations,
   shapes, systematic matrix layout, and invalid-input behavior. Test finite
   fields first, then linear/cyclic codes, BCH/RS, and channel calculations using
   independent answers, algebraic properties, and exhaustive small cases.
3. **Bug fixes — finite-field element fixes awaiting review.** Correct exponent
   normalization, division by zero, and powers of zero. Other fixes remain.
4. **Formatting and linting.** Apply Ruff in a separate change.
5. **Package structure.** Move implementation into a package, preserving old
   imports with compatibility modules and keeping `playground.py` runnable.

## Baseline and validation

- Python requirement: 3.10 or newer; local validation currently covers only
  Python 3.13.
- NumPy is temporarily constrained to 2.2.x. The resolved baseline uses 2.2.6.
- Run `uv run python -m pytest` for the initial seven tests. They protect six
  existing imports and selected numerical results in the two printed examples;
  they do not establish correctness across all algorithms or inputs.
- [Finite-field contract](docs/finite-field-interface.md) and
  `tests/test_galois_field.py` cover element arithmetic in GF(2), GF(4), GF(8),
  and GF(16) using an independent binary-polynomial reference. The fixes remove
  all expected-failure markers and restore exhaustive division and negative
  power checks. Current local result: **64 passed, no expected failures**;
  Ruff checks pass for the test files.
- The project runs from the repository root without being installed as a package.
- `uv run ruff check .` and `uv run ruff format --check .` expose existing source
  issues; passing them across the codebase is deferred to step 4.

## Confirmed issues to address

- With NumPy 2.5.4, `playground.py` fails because `intToArray()` uses the removed
  binary mode of `np.fromstring`. Fix this and reconsider the NumPy constraint.
- `BSC.ChannelEfficiency()` references `bsc` rather than `self`, causing a
  `NameError` when imported and called normally.

## Fixes awaiting review

- Normalize finite exponents modulo q - 1 before converting to labels. This
  fixes `elementFromExp(-1)`, division with exponent difference -1, and alpha^-1.
- Raise `ZeroDivisionError` for division by zero, including 0/0.
- Return 1 for `powElement(0, 0)`, consistent with the reviewed contract and
  polynomial evaluation; raise `ZeroDivisionError` for negative powers of zero.

The README also spells GF(2^5) as GF(25) in its first exercise; correct that when
reviewing the mathematical documentation.
