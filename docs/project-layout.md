# Project layout and compatibility

Algorithm implementations live in the `coding_theory_algorithms` package:

| Previous module | Implementation module |
| --- | --- |
| `GaloisField` | `coding_theory_algorithms.galois_field` |
| `LinearBlockCode` | `coding_theory_algorithms.linear_block_code` |
| `CyclicCode` | `coding_theory_algorithms.cyclic_code` |
| `BCHCode` | `coding_theory_algorithms.bch_code` |
| `RSCode` | `coding_theory_algorithms.rs_code` |
| `BinarySymmetricChannel` | `coding_theory_algorithms.binary_symmetric_channel` |

Each root module forwards to its implementation's module object. Existing
classes, functions, constants, singleton fields, and module attribute changes
are therefore shared across both import paths. This avoids duplicate field
instances and preserves monkeypatching through the old names. Introspection
such as `__file__` and class `__module__` points to the new implementation.

Both import styles work from the repository root:

```python
from GaloisField import GaloisField
from coding_theory_algorithms.cyclic_code import CyclicCode
```

`playground.py` retains its existing algorithm imports and runs with
`python3 playground.py`. The channel demonstration works with both
`python3 BinarySymmetricChannel.py` and
`python3 -m coding_theory_algorithms.binary_symmetric_channel`.
Other root module files also preserve script execution through forwarding.

The package initializer keeps imports explicit in submodules. Internal
algorithm dependencies use relative imports and do not depend on the root
compatibility files. Tests remain in `tests/` and interface documentation in
`docs/`.

This change introduces a source package. Dependency installation remains
`uv sync --locked`; the project is not installed into the environment as a
distribution (`tool.uv.package = false`). Build/distribution configuration is
outside this step.

Verification compares moved implementations' ASTs with the committed source,
allowing only package-relative import changes. Compatibility tests run in fresh
Python processes without inherited `PYTHONPATH`, checking both import orders,
class identity, shared module attributes and GF2, and the script results. The
existing mathematical suite continues to use the legacy imports.
