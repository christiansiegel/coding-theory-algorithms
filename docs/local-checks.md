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

`uv python dir` reports the managed interpreter directory (on the validation
host, `/home/cs/.local/share/uv/python`). These interpreters live outside the
project `.venv`. The compatibility target disables automatic Python downloads
and uses isolated environments without replacing the working `.venv`.
Python installation may also create versioned executables in uv's executable
directory; it does not replace the system Python.

Dependencies are exported from `uv.lock` with Python-version markers. The
matrix tests the locked NumPy 2.2.x floor on Python 3.10 and newer NumPy builds
on later Pythons. It uses two workers and reports each interpreter/dependency
version and test result; any failed run gives a nonzero exit status.

Use `make compatibility PYTHONS="3.13"` for one installed version. Override `UV`
to select a different uv executable. All checks are local; no CI, hooks, or
remote service is configured.

## Verified matrix

| Python | Locked NumPy | Full suite |
| --- | --- | --- |
| 3.10.21 | 2.2.6 | Passed |
| 3.11.16 | 2.4.6 | Passed |
| 3.12.14 | 2.5.4 | Passed |
| 3.13.5 | 2.5.4 | Passed |
| 3.14.7 | 2.5.4 | Passed |

The full matrix checked 514 tests. The final channel validation and README
fixture changes were then checked with 143 focused tests on each interpreter.
The final Python 3.13 suite has 517 tests. These results cover the locked
dependency combinations above, rather than every Python/NumPy combination.
