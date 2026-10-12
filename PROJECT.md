# Code quality work

## Current status

The planned correctness, tooling, package, API, compatibility, and documentation
work is complete. Changes were committed in small steps; the final batch was
authorized for commits without review. Git history records the individual fixes.

- Independent known-answer, algebraic-property, and exhaustive small-case tests
  cover finite fields, polynomials, binary codes, BCH/RS decoding, channels,
  probability calculations, and diagnostic helpers.
- Confirmed defects were fixed, including field exponents, polynomial division,
  channel capacity, cyclic distance, BCH/RS decoding, syndrome diagnostics,
  conjugate roots, and incomplete standard arrays.
- Public constructors, code vectors, systematic matrices, and probabilities
  have explicit validation. RS rejects inherited binary-only operations.
- Implementations live in `coding_theory_algorithms/`; legacy root import shims
  were removed. Descriptive APIs use snake_case, while lecture notation such as
  `c()`, `s()`, `G()`, `H()`, and `PU()` remains available.
- Ruff lint and formatting use a 200-character line length. Playground imports
  intentionally make the algorithms available for experimentation.
- Make provides local checks and a locked compatibility matrix. No remote CI
  or automatic hooks are configured.
- README solutions were regenerated from the current playground, including the
  complete 16-row standard array for its (7,3) code.

## Validation

The full 514-test suite passed on Python 3.10.21, 3.11.16, 3.12.14, 3.13.5,
and 3.14.7, with the NumPy versions selected by `uv.lock`. Final channel-type
regressions add three tests; all 143 validation and linear-code tests also passed
on each interpreter after those final changes. Final `make check` passed all
517 tests, lint, and formatting.

See [local checks](docs/local-checks.md) for commands and interpreter storage.
Compatibility results describe tested versions; the Python >=3.10 declaration
does not promise compatibility with future Python or NumPy releases.

## Reference documents

- [API migration](docs/api-migration.md) and [project layout](docs/project-layout.md)
- [Input validation](docs/input-validation.md)
- [Finite fields](docs/finite-field-interface.md) and [polynomials](docs/polynomial-interface.md)
- [Binary codes](docs/binary-code-interface.md) and [BCH/RS](docs/bch-rs-interface.md)
- [Channels](docs/channel-interface.md) and [Euclidean helpers](docs/euclidean-interface.md)

## Remaining scope

Low-level polynomial helpers assume valid coefficient representations as
documented. Decoding guarantees apply within the stated correction radius;
behavior beyond it is not a general error-detection guarantee. Exact distance
and exhaustive tables remain exponential, appropriate for small textbook cases.
Performance profiling should precede any further optimization or broad refactor.
