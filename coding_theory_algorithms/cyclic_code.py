import numpy as np
from .galois_field import GF2, X, degree
from .linear_block_code import LinearBlockCode
from .validation import binary_remainder, coefficients, integer

"""
IMPORTANT:
When NumPy creates a polynomial from an array, it uses the highest index
array element as coefficient with the lowest position and vice versa.
Thus we flip every array before converting it to a polynomial!

Example:
 Default NumPy behaviour:    [0,1,0,1] -> X^2 + 1
 What we want:               [0,1,0,1] -> X + X^3

Moreover all function take coefficient arrays instead of Numpy polynomial
objects as parameters.
"""


def shift(v, i=1):
    """Cyclic shift of a polynomial.
    (slide 3)
    Args:
        v: polynomial to shift
        i: shift polynomial i times (default: 1)
    Returns:
        Polynomial v shiftet for i times.
    """
    return np.roll(v, i)


def encode(m, g, systematic=True):
    """Encoding of cyclic code (in systematic form)
    (slide 23)
    ATTENTION: Dangling zeros in returned codeword are cut away.
    """
    if systematic:
        r = degree(g)  # r = n - k
        Xr = X(r)  # X^(n-k)
        XrmX = GF2.mult_poly(Xr, m)  # X^(n-k) * m(X)
        p = GF2.mod_poly(XrmX, g)  # p(X) = (X^(n-k) * m(X)) mod g(X)
        c = GF2.add_poly(p, XrmX)  # c(X) = p(X) + (X^(n-k) * m(X))
    else:
        c = GF2.mult_poly(m, g)
    return c.astype(int)


def gToG(g, n, systematic=True, verbose=False):
    """Builds Generator Matrix G from given Generator Polynomial.
    (slide 25)
    Args:
        g: generator polynomial
        n: code length
        systematic: generator matrix in systematic form (default: True)
        verbose: verbose output on how to gain systematic form (default: False)
    Returns:
        Generator Matrix
    """
    k = n - degree(g)
    g = pad_end(g, n)
    G = np.empty([k, n])
    for i in range(0, k):
        G[i, :] = shift(g, i)

    # row additions to gain embedded identity matrix on right side
    # -> systematic form
    if systematic:
        G = make_systematic(G, verbose)

    return G.astype(int)


def make_systematic(G, verbose=True):
    G = np.asarray(G)
    if G.ndim != 2 or not 0 < G.shape[0] < G.shape[1]:
        raise ValueError("G must have shape (k,n) with 0 < k < n")
    G = coefficients(G.reshape(-1), "G").reshape(G.shape)
    k, n = G.shape
    if verbose:
        print("unsystematic:")
        print(G.astype(int))
        print()

    for i in range(k):
        column = n - k + i
        pivots = np.flatnonzero(G[i:, column])
        if not pivots.size:
            raise ValueError("The rightmost k columns must be invertible; column permutations are not performed")
        pivot = i + int(pivots[0])
        G[[i, pivot]] = G[[pivot, i]]
        for row in range(k):
            if row != i and G[row, column]:
                G[row] ^= G[i]
        if verbose:
            print("Pivot in column", column)
            print(G.astype(int))
            print()
    return G.astype(int)


def print_all_cyclic_codes(factorPolynomials):
    """Generates all cyclic codes that can be created from
    the given factor polynomials.
    (slide 28)
    Args:
        factorPolynomials: factor polynomials in a python array
    """
    s = ""
    product = np.array([])
    for i in range(0, len(factorPolynomials)):
        if i == 0:
            product = factorPolynomials[i]
        else:
            product = GF2.mult_poly(product, factorPolynomials[i])
        s += "(" + GF2.poly_to_string(factorPolynomials[i]) + ") "
    print(s + "= " + GF2.poly_to_string(product))
    print()

    numberCodes = 2 ** (len(factorPolynomials)) - 2
    n = degree(product)
    print("There are", numberCodes, "different cyclic codes of length", n, "as")
    print("we can find", numberCodes, "different generator polynomials that are")
    print("the factors of", GF2.poly_to_string(product))

    print("Code <- Generator polynomial")
    for i in range(0, numberCodes):
        s = ""
        gp = np.array([])  # generator polynomial
        for j in range(0, len(factorPolynomials)):
            if np.bitwise_and(i + 1, 2**j) > 0:
                if s == "":
                    gp = factorPolynomials[j]
                else:
                    gp = GF2.mult_poly(gp, factorPolynomials[j])
                s += "(" + GF2.poly_to_string(factorPolynomials[j]) + ")"

        print("Ccyc(" + str(n) + ", " + str(n - degree(gp)) + ") <- g" + str(i + 1) + " = " + s + " = " + GF2.poly_to_string(gp))


def pad_end(p, length):
    assert p.size <= length, "pad_end() failed because polynomial is longer than given size."

    p = np.pad(p, (0, length - p.size), "constant", constant_values=0)
    return p


class CyclicCode(LinearBlockCode):
    """Cyclic Code

    Based on the the Cyclic Codes lecture (2016-18-02).

    Attributes:
        _g: The Generator Polynomial of the Cyclic Code
        _n: Code length
    """

    _g = np.empty([0])
    _n = 0

    def __init__(self, g, n):
        n = integer(n, "n", minimum=2)
        g = coefficients(g, "g")
        g = g[: degree(g) + 1]
        mask = sum(int(bit) << i for i, bit in enumerate(g))
        if g[0] != 1 or not 0 < degree(g) < n or binary_remainder((1 << n) | 1, mask):
            raise ValueError("g must be a nonconstant binary divisor of X^n+1 with degree less than n")
        self._g = g.copy()
        self._n = n

    def g(self):
        return self._g.astype(int)

    def print_g(self):
        print(GF2.poly_to_string(self.g()))

    def n(self):  # override
        return self._n

    def k(self):  # override
        return self.n() - degree(self.g())

    def dmin(self, verbose=False):  # override
        """Exact minimum distance by enumerating nonzero codewords.
        Generator weight is only an upper bound. Enumeration is exponential
        in k and intended for small textbook examples.
        """
        # Enumerate using matrix encoding rather than repeating polynomial
        # division for every message. Systematic form is unnecessary here.
        linear_code = LinearBlockCode()
        linear_code.set_g(self.G())
        return linear_code.dmin(verbose)

    def dmin_verbose(self):
        self.dmin(True)

    def G(self, systematic=True, verbose=False):  # override
        return gToG(self.g(), self.n(), systematic, verbose)

    def set_g(self, *args):  # override
        raise NotImplementedError("Set the generator through the cyclic-code constructor")

    def set_h(self, *args):  # override
        raise NotImplementedError("Set the generator through the cyclic-code constructor")

    def shift(self, c, i=1):
        """Cyclic right shift of c using division (slide 11)"""
        Xi = X(i)  # X^i polynomial
        XiCX = GF2.mult_poly(Xi, c)  # X^i * c(X) polynomial
        Xn1 = GF2.add_poly(X(self.n()), X(0))  # X^n + 1 polynomial
        ci = GF2.mod_poly(XiCX, Xn1)  # i times shifted c
        ci = ci[: degree(ci) + 1]  # remove cancelled high coefficients before padding
        return pad_end(ci, self.n())

    def c(self, m, systematic=True):  # override
        """encode message polynomial m
        Args:
            m: message polynomial
            systematic: return codeword in systematic form (default: True)
        Returns:
            codeword
        """
        m = coefficients(m, "m", maximum_length=self.k())
        c = encode(m, self.g(), systematic)
        return pad_end(c, self.n())

    def print_message_codeword_table(self, systematic=True):  # override
        """Print all messages and their corresponding codewords.
        Args:
            systematic: print codewords in systematic form (default: True)
        """
        M = self.M()
        print("Messages -> Codewords")
        for m in M:
            c = self.c(m, systematic)
            print(m, c, "m(X) =", GF2.poly_to_string(m), "\tc(X) =", GF2.poly_to_string(c))

    def S(self, r):
        """Calculate Syndrome polynomial from receive or error polynomial.
        Args:
            r: receive or error polynomial
        Returns:
            Syndrome polynomial
        """
        return GF2.mod_poly(r, self.g())

    def shift_syndrome(self, S, i=1):
        """Shift syndrome i times (slide 35)"""
        for i in range(0, i):
            # S1(X) = XS(X) mod g(X)
            S = GF2.mod_poly(GF2.mult_poly(X(1), S), self.g())
        return S
