# Project layout

Implementations live in `coding_theory_algorithms/`. Import explicit submodules:

```python
from coding_theory_algorithms.galois_field import GaloisField, GF2
from coding_theory_algorithms.cyclic_code import CyclicCode
```

Root import shims have been removed. See the [migration guide](api-migration.md)
for the intentional import and descriptive-name API breaks. Classes and lecture
notation remain available in their implementation modules.

`playground.py` keeps the algorithm imports ready for additional exercises.
Run it from the repository root with `uv run python playground.py`. Run the
channel demonstration with
`uv run python -m coding_theory_algorithms.binary_symmetric_channel`.

Internal imports are relative. The package initializer avoids eager imports;
the GF2 singleton is shared by dependent modules. Tests live in `tests/`,
interface contracts in `docs/`, and local compatibility tooling in `scripts/`.

Install dependencies with `uv sync --locked`. This remains a repository-local
source package (`tool.uv.package = false`); building/publishing a distribution
is not part of this project workflow.
