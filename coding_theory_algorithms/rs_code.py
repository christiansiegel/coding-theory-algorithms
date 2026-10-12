# -*- coding: UTF-8 -*-
import numpy as np
from .bch_code import BCHCode
from .cyclic_code import pad_end
from .galois_field import GaloisField, X, degree
from .validation import coefficients, integer


def g(GF, t, verbose=True):
    """Construct generator polynomial of RS code with t.
    (p 116)
    Args:
        GF: ExtendedGaloisField
        t: number of correctable errors
        verbose: printout how the generator polynomial is generated
    Returns:
        Generator polynomial of BCH code with t
    """
    if verbose:
        print("")
        print("Generator polynomial g(X) of BCH code with t = " + str(t) + ":")
        print("")
        print("g(X) = (X - \u03b1)(X - \u03b1^2)...(X - \u03b1^(n-k))")
        print("")

    g = np.ones(1)
    for i in range(1, 2 * t + 1):  # n - k = 2*t
        root = GF.element_from_exp(i)
        phi = np.array([root, 1])
        g = GF.mult_poly(g, phi)
        if verbose:
            print("\u03a6_" + str(i) + "(X) = " + GF.poly_to_string(phi))

    if verbose:
        print("g(X) = " + GF.poly_to_string(g))
        print()

    return g


class RSCode(BCHCode):
    """RS Code

    Attributes:
        _GF: Galois Field the generator polynomial has roots in.
    """

    _GF = None
    _t = 0

    def __init__(self, GF, t, verbose=False):
        """
        Args:
            GF: Galois Field the generator polynomial has roots in.
            t: Number of errors to correct.
            verbose: Print how the generator polynomial is constructed.
        """
        if not isinstance(GF, GaloisField):
            raise ValueError("GF must be a GaloisField instance")
        t = integer(t, "t", minimum=1)
        if 2 * t >= GF.q() - 1:
            raise ValueError("RS parameters require 2t < n=q-1")
        m = GF.m()
        self._alphabet = GF.q()
        self._GF = GF
        self._g = g(GF, t, verbose)
        self._n = 2**m - 1  # slide 5
        self._t = t

    def GF(self):
        """Return Galois Field."""
        return self._GF

    def c(self, m, systematic=True):
        """Encode a message polynomial over the supplied field, padded to n symbols."""
        m = coefficients(m, "m", alphabet=self._alphabet, maximum_length=self.k())
        GF = self.GF()
        if systematic:
            shifted = GF.mult_poly(X(degree(self.g())), m)
            remainder = GF.mod_poly(shifted, self.g())
            codeword = GF.add_poly(shifted, remainder)
        else:
            codeword = GF.mult_poly(m, self.g())
        return pad_end(codeword.astype(int), self.n())

    def dmin(self, verbose=False):
        """Minimum symbol distance of a primitive RS code with valid parameters."""
        distance = 2 * self._t + 1
        if verbose:
            print("Minimum symbol distance: dmin = 2t + 1 = " + str(distance))
        return distance

    def t(self):
        """Return the designed symbol-error correction radius."""
        return self._t

    def _binary_only(self, *args, **kwargs):
        raise NotImplementedError("This binary-code method is unsupported for Reed-Solomon symbols; use c(), S(), and decode().")

    G = H = P = s = M = C = _binary_only
    Ai = A = PU = Pe = _binary_only
    shift = shift_syndrome = _binary_only
    correctable_error_patterns = decoding_table = syndrome_decode = _binary_only
    print_message_codeword_table = print_parity_check_equations = print_syndrome_vector_equations = _binary_only
    print_errors_that_have_syndrome = print_standard_array = print_decoding_table = verbose_syndrome_decode = _binary_only

    def print_g(self):
        print(self.GF().poly_to_string(self.g()))

    def print_info(self):
        """Print the generator's actual roots and parameters measured in symbols."""
        GF = self.GF()
        t = self._t
        print()
        print("Generator Polynomial:        g(X) = " + GF.poly_to_string(self.g()))
        rootStr = ", ".join(GF.element_to_string(root) for root in GF.roots(self.g()))
        print("Roots of g(X) in GF(2^" + str(GF.m()) + "):    " + rootStr)
        print("Code length (symbols):       n = q - 1 = " + str(self.n()))
        print("Message length (symbols):    k = " + str(self.k()))
        print("Number of parity symbols:    n - k = 2t = " + str(2 * t))
        print("Minimum symbol distance:     dmin = " + str(2 * t + 1) + " = 2t + 1")
        print("Error correction capability (symbols): t = " + str(t))
        print()
