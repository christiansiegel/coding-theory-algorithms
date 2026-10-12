# Keep all algorithm imports ready for experimenting with other exercises.
# ruff: noqa: F401
import numpy as np
from coding_theory_algorithms.binary_symmetric_channel import BSC
from coding_theory_algorithms.galois_field import GaloisField
from coding_theory_algorithms.galois_field import GF2
from coding_theory_algorithms.linear_block_code import LinearBlockCode
from coding_theory_algorithms.cyclic_code import CyclicCode
from coding_theory_algorithms.bch_code import BCHCode

print("Exercise 1")
"""
A binary linear cyclic code Ccyc(n, k) has code length n = 7 and generator polynomial 
g(X) = 1 + X2 + X3 + X4.

(a) Find the code rate, the generator and parity check matrices of the code in systematic form, and its Hamming distance.
(b) If all the information symbols are ‘1’s, what is the corresponding code vector?
(c) Find the syndrome corresponding to an error in the first information symbol, and show that the code is capable of correcting this error.
"""
p = np.array([1, 0, 1, 0, 0, 1])
GF25 = GaloisField(p)
GF25.printInfo()

print("Exercise 2")
"""
A binary linear cyclic code Ccyc(n, k) has code length n = 7 and generator polynomial 
g(X) = 1 + X2 + X3 + X4.

(a) Find the code rate, the generator and parity check matrices of the code in systematic form, and its Hamming distance.
(b) If all the information symbols are ‘1’s, what is the corresponding code vector?
(c) Find the syndrome corresponding to an error in the first information symbol, and show that the code is capable of correcting this error.
"""
g = np.array([1, 0, 1, 1, 1])
cc = CyclicCode(g, 7)
cc.printInfo()
