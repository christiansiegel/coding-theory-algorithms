# -*- coding: UTF-8 -*-
import numpy as np
from .cyclic_code import CyclicCode
from .galois_field import GaloisField, X, degree
from .validation import coefficients, integer


def g(GF, t, verbose=True):
    """Construct generator polynomial of BCH code with t.
    (slide 8)
    Args:
        GF: ExtendedGaloisField the roots are taken from
        t: number of correctable errors
        verbose: printout how the generator polynomial is generated
    Returns:
        Generator polynomial of BCH code with t
    """
    i_list = list(range(1, (2 * t) + 1, 2))  # only odd roots 1 <= i <= 2t

    if verbose:
        print("")
        print("Generator polynomial g(X) of BCH code with t = " + str(t) + ":")
        print("")
        print("Assuming \u03b1 is a primitive element in GF(2^" + str(GF.m()) + "), ")
        print("the generator polynomial g(X) has the odd roots: \u03b1^i, i = 1, 3, ..., 2t")
        print("Assuming \u03a6i(X) is the minimal polynomial of \u03b1^i, then")
        print("g(X) can be expressed by the lowest common multiple (LCM) of the")
        print("minimal polynomials:")
        print("")
        # g(X) = LCM{ϕ_1(X), ϕ_2(X), . . . , ϕ_2t(X)}
        s = "g(X) = LCM{"
        for i in i_list:
            s += "\u03a6_" + str(i) + "(X), "
        s = s[:-2] + "}"
        print(s)

    g = np.ones(1)
    included_roots = set()
    for i in i_list:
        root = GF.element_from_exp(i)
        if root in included_roots:
            continue  # the minimal polynomial for this conjugacy class is included
        conjugate_roots = GF.conjugate_roots(root)
        included_roots.update(conjugate_roots)
        phi = GF.minimal_polynomial(conjugate_roots)
        g = GF.mult_poly(g, phi)
        if verbose:
            print("\u03a6_" + str(i) + "(X) = " + GF.poly_to_string(phi))

    if verbose:
        print("g(X) = " + GF.poly_to_string(g))
        print()

    return g


def HCF(A, B):
    """Print integer Euclid steps and return the nonnegative GCD exactly."""
    A, B = abs(A), abs(B)
    if A < B:
        A, B = B, A
    if A == 0:
        print(0)
        return 0

    print(-1, "\t", A, "\t", "-", "\t", 1, "\t", 0)
    if B != 0:
        print(0, "\t", B, "\t", "-", "\t", 0, "\t", 1)
    previous_r, r = A, B
    previous_s, s = 1, 0
    previous_t, t = 0, 1
    i = 1
    while r != 0:
        quotient = previous_r // r
        next_r = previous_r - quotient * r
        next_s = previous_s - quotient * s
        next_t = previous_t - quotient * t
        if next_r != 0:
            print(i, "\t", next_r, "\t", quotient, "\t", next_s, "\t", next_t)
        previous_r, r = r, next_r
        previous_s, s = s, next_s
        previous_t, t = t, next_t
        i += 1
    print(previous_r)
    return previous_r


class BCHCode(CyclicCode):
    """BCH Code

    Based on the the BCH Codes lecture (2016-25-02).

    Attributes:
        _GF: Galois Field the generator polynomial has roots in.
    """

    _GF = None
    _alphabet = 2

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
            raise ValueError("BCH parameters require 2t < n=q-1")
        m = GF.m()
        self._GF = GF
        self._g = g(GF, t, verbose)
        self._n = 2**m - 1  # slide 5
        if degree(self._g) >= self._n:
            raise ValueError("BCH generator must leave a positive message dimension")

    def GF(self):
        """Return Galois Field."""
        return self._GF

    def H(self, systematic=True):
        """Return paritiy check matrix H.
        Args:
            systematic: If true return H in systematic form.
                        Else use method from slide 15 to generate H.
        """
        if systematic:
            # use inherited function
            return super(BCHCode, self).H()

        # else use method on slide 15:
        n = self.n()
        t = self.t()
        H = None
        for ni in range(0, n):  # 0,1,2...n-1
            Hcol = None  # ni'th column vector of H
            for ti in range(1, 2 * t, 2):  # 1,3,5...2t-1
                alpha = self.GF().element_from_exp(ti * ni)  # (a^ti)^ni
                alpha = self.GF().element_to_poly(alpha, size_m=True)[:, np.newaxis]
                if Hcol is None:
                    Hcol = alpha
                else:
                    Hcol = np.concatenate((Hcol, alpha), axis=0)  # below
            if H is None:
                H = Hcol
            else:
                H = np.concatenate((H, Hcol), axis=1)  # right
        return H.astype(int)

    def S(self, r, verbose=True):  # override
        """Calculate Syndrome polynomial from receive or error polynomial.
        (slide 17, 26)
        Args:
            r: receive or error polynomial
        Returns:
            Syndrome polynomial
        """
        r = coefficients(r, "r", alphabet=self._alphabet, length=self.n())
        if verbose:
            print()
            print("The syndrome vector components are:")

        t = self.t()
        GF = self.GF()
        S = np.ones(2 * t)

        for i in range(1, 2 * t + 1):  # 1 <= i <= 2t
            exp_a = i
            a = GF.element_from_exp(exp_a)
            s = GF.substitute_element_into_poly(r, a)
            S[i - 1] = s

            if verbose:
                syndromeVectorStr = "s_" + str(i) + " = r(" + GF.element_to_string(a) + ") = "
                syndromeVectorStr += GF.poly_to_string(r, "(" + GF.element_to_string(a) + ")") + " = "
                syndromeVectorStr += GF.element_to_string(s)
                print(syndromeVectorStr)

        if verbose:
            print()
            print("Therefore the syndrome polynomial is:")
            print("S(X) =", GF.poly_to_string(S))
            print()

        return S.astype(int)

    def decode(self, r, verbose=True):
        """Decode received polynomial r(X) using the Euclidean Algorithm.
        (slide 26)
        Args:
            r: received polynomial
            verbose: print the algorithm steps.
        """
        r = coefficients(r, "r", alphabet=self._alphabet, length=self.n())
        GF = self.GF()

        if verbose:
            print()
            print("Decode the received polynomial:")
            print("r(X) = " + self.GF().poly_to_string(r))

        S = self.S(r, verbose)
        if np.count_nonzero(S) == 0:
            if verbose:
                print("All syndrome components are zero; returning the received codeword.")
            return r.astype(int).copy()

        if verbose:
            print("The Euclidean algorithm is applied by constructing the " + "following table:")

        ri, ti = GF.HCF(X(2 * self.t()), S, verbose)

        lamb = GF.monic_multiplier(ti)

        errorLocationPoly = GF.mult_poly(lamb, ti)
        errorLocationPolyDerivative = GF.derive_poly(errorLocationPoly)
        errorEvalutationPoly = GF.mult_poly(lamb, ri)
        if verbose:
            print("An element \u03bb \u2208 GF(" + str(GF.q()) + ") is conveniently selected to multiply")
            print("t_i(X) by, in order to convert it into a monic polynomial.")
            print("This value of \u03bb is \u03bb = " + GF.element_to_string(lamb))
            print("Therefore:")
            print()
            print("Error location polynomial:")
            print("\u03c3(X) = \u03bb * t_i(X) = " + GF.element_to_string(lamb) + "(" + GF.poly_to_string(ti) + ")")
            print("                  = " + GF.poly_to_string(errorLocationPoly))
            print()
            print("\u03c3'(X) = " + GF.poly_to_string(errorLocationPolyDerivative))
            print()
            print("Error evaluation polynomial:")
            print("W(X) = -\u03bb * r_i(X) = " + GF.element_to_string(lamb) + " * " + GF.poly_to_string(ri))
            print("                   = " + GF.poly_to_string(errorEvalutationPoly))
            print()
            print("Performing Chien search in the error location polynomial \u03c3(X):")
            print()

        errorLocations = []
        for i, root in enumerate(GF.roots(errorLocationPoly)):
            j = GF.element_to_exp(GF.element_from_exp(-GF.element_to_exp(root)))
            errorLocations.append(j)
            if verbose:
                print("\u03b1^(-j_" + str(i + 1) + ") = " + GF.element_to_string(root) + "\t-> j_" + str(i + 1) + " = " + str(j))

        if verbose:
            print("\nError values:")

        errorValues = []

        for i, errorLocation in enumerate(errorLocations):
            alpha = GF.element_from_exp(-errorLocation)
            res_W = GF.substitute_element_into_poly(errorEvalutationPoly, alpha)
            res_o = GF.substitute_element_into_poly(errorLocationPolyDerivative, alpha)
            errorValue = GF.div_elements(res_W, res_o)
            errorValues.append(errorValue)
            if verbose:
                print(
                    "e_j"
                    + str(i + 1)
                    + " = W("
                    + GF.element_to_string(alpha)
                    + ") / \u03c3'("
                    + GF.element_to_string(alpha)
                    + ") = "
                    + GF.element_to_string(res_W)
                    + " / "
                    + GF.element_to_string(res_o)
                    + " = "
                    + GF.element_to_string(errorValue)
                )

        e = np.zeros(self.n(), dtype=int)
        for i, errorLocation in enumerate(errorLocations):
            errorValue = errorValues[i]
            e[errorLocation] = errorValue

        c = GF.add_poly(r, e)

        if verbose:
            print()
            print("Error polynomial:")
            print("e(X) = " + GF.poly_to_string(e))
            print()
            print("Code vector:")
            print("c = r + e = " + str(c))

        return c.astype(int)

    def print_info(self):
        """
        (slide 5, 7)
        """
        GF = self.GF()
        print()
        print("Generator Polynomial:        g(X) = " + GF.poly_to_string(self.g()))
        rootStr = ""
        roots = GF.remove_conjugate_roots(GF.roots(self.g()))
        for root in roots:
            rootStr += GF.element_to_string(root) + ", "
        rootStr = rootStr[:-2]  # remove last comma
        print("Roots of g(X) in GF(2^" + str(self.GF().m()) + "):    " + rootStr + " and all conjugate roots")
        print("Code length:                 n = 2^m - 1 = " + str(self.n()))
        print("Message length:              k = " + str(self.k()))
        print("Number of parity bits:       n - k = " + str(self.n() - self.k()) + " <= mt")
        print("Minimum Hamming distance:    dmin = " + str(self.dmin()) + " >= 2t + 1")
        print("Error correction capability: t = " + str(self.t()))
        print()
