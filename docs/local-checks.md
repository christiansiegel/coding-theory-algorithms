# Local checks

Run `uv sync --locked`, then `make check` (or simply `make`) for Ruff lint,
format checks, and the complete pytest suite. `make test`, `make lint`, and
`make format-check` run individual checks; `make format` changes formatting.

The project declares Python >=3.10. The compatibility matrix targets Python
3.10, 3.11, 3.12, 3.13, and 3.14. Install interpreters explicitly, then run:

```sh
uv python install 3.10 3.11 3.12 3.13 3.14
make compatibility
```

`uv python dir` reports where uv stores managed interpreters, outside the
project `.venv`. The compatibility target disables automatic Python downloads
and uses isolated environments without replacing the working `.venv`.

Dependencies are exported from `uv.lock` with Python-version markers. The
matrix checks the dependency combination selected for each interpreter. It uses
two workers and reports versions and test results; any failed run gives a
nonzero exit status.

Use `make compatibility PYTHONS="3.13"` for one installed version. Override `UV`
to select a different uv executable.

The suite uses independent known answers, algebraic properties, and exhaustive
small cases. Checks are intended for textbook-sized inputs.
