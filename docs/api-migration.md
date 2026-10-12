# API migration

This migration intentionally removes the six root import modules and old
descriptive method names. No deprecated aliases are provided for those names.

| Former import | Current module |
| --- | --- |
| GaloisField | coding_theory_algorithms.galois_field |
| LinearBlockCode | coding_theory_algorithms.linear_block_code |
| CyclicCode | coding_theory_algorithms.cyclic_code |
| BCHCode | coding_theory_algorithms.bch_code |
| RSCode | coding_theory_algorithms.rs_code |
| BinarySymmetricChannel | coding_theory_algorithms.binary_symmetric_channel |

Class names remain unchanged. For example:

```python
from coding_theory_algorithms import LinearBlockCode

code = LinearBlockCode()
code.set_g([[1, 1, 1]])
codeword = code.c([1])
syndrome = code.s(codeword)
code.print_standard_array()
```

The main classes and `GF2` are also exported directly from the package.
Explicit submodule imports remain supported for classes and utility functions.

Lecture/book notation is retained: c, s, G, H, P, M, C, R, S, A, Ai, PU, Pe,
Cs, PX, PY, PXY, PYX, HX, HY, HXY, HYX, GF, GF2, X, HCF, nCr, and matrix
conversion notation GtoH, HtoG, GtoP, HtoP, gToG. The dmin keyword argument is
now `verbose` rather than `Verbose`.

Descriptive APIs use snake_case:

| Former name | Current name |
| --- | --- |
| `ChannelEfficiency` | `channel_efficiency` |
| `addElements` | `add_elements` |
| `addPoly` | `add_poly` |
| `arrayToString` | `array_to_string` |
| `conjugateRootGroups` | `conjugate_root_groups` |
| `conjugateRoots` | `conjugate_roots` |
| `constructGF` | `construct_gf` |
| `correctableErrorPatterns` | `correctable_error_patterns` |
| `decodingTable` | `decoding_table` |
| `derivePoly` | `derive_poly` |
| `divElements` | `div_elements` |
| `divPoly` | `div_poly` |
| `divmodPoly` | `divmod_poly` |
| `dminOfPoly` | `dmin_of_poly` |
| `dminVerbose` | `dmin_verbose` |
| `elementFromExp` | `element_from_exp` |
| `elementFromPoly` | `element_from_poly` |
| `elementToExp` | `element_to_exp` |
| `elementToPoly` | `element_to_poly` |
| `elementToString` | `element_to_string` |
| `elementsAsPoly` | `elements_as_poly` |
| `errorDetectionCapability` | `error_detection_capability` |
| `intToArray` | `int_to_array` |
| `isFactor` | `is_factor` |
| `makeSystematic` | `make_systematic` |
| `matrixMultiplicationEquations` | `matrix_multiplication_equations` |
| `minimalPolynomial` | `minimal_polynomial` |
| `modPoly` | `mod_poly` |
| `monicMultiplier` | `monic_multiplier` |
| `multElements` | `mult_elements` |
| `multPoly` | `mult_poly` |
| `padEnd` | `pad_end` |
| `polyToString` | `poly_to_string` |
| `powElement` | `pow_element` |
| `printAllCyclicCodes` | `print_all_cyclic_codes` |
| `printDecodingTable` | `print_decoding_table` |
| `printErrorsThatHaveSyndrome` | `print_errors_that_have_syndrome` |
| `printInfo` | `print_info` |
| `printMessageCodewordTable` | `print_message_codeword_table` |
| `printMinimalPolynomials` | `print_minimal_polynomials` |
| `printParityCheckEquations` | `print_parity_check_equations` |
| `printStandardArray` | `print_standard_array` |
| `printSyndromeVectorEquations` | `print_syndrome_vector_equations` |
| `printg` | `print_g` |
| `removeConjugateRoots` | `remove_conjugate_roots` |
| `setG` | `set_g` |
| `setH` | `set_h` |
| `shiftSyndrome` | `shift_syndrome` |
| `substituteElementIntoPoly` | `substitute_element_into_poly` |
| `syndromeDecode` | `syndrome_decode` |
| `verboseSyndromeDecode` | `verbose_syndrome_decode` |
