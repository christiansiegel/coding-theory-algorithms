import numpy as np
import LinearBlockCode as lbc
from GaloisField import GF2, X, degree
from LinearBlockCode import LinearBlockCode

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

def shift(v, i = 1):
    """Cyclic shift of a polynomial.
    (slide 3)
    Args:
        v: polynomial to shift
        i: shift polynomial i times (default: 1)
    Returns:
        Polynomial v shiftet for i times.
    """
    return np.roll(v, i)

def encode(m, g, systematic = True):
    """Encoding of cyclic code (in systematic form)
    (slide 23)
    ATTENTION: Dangling zeros in returned codeword are cut away.
    """
    if systematic:
        r = degree(g) # r = n - k
        Xr = X(r) # X^(n-k)
        XrmX = GF2.multPoly(Xr, m) # X^(n-k) * m(X)
        p = GF2.modPoly(XrmX, g) # p(X) = (X^(n-k) * m(X)) mod g(X)
        c = GF2.addPoly(p, XrmX) # c(X) = p(X) + (X^(n-k) * m(X))
    else:
        c = GF2.multPoly(m, g)
    return c.astype(int)

def gToG(g, n, systematic = True, verbose = False):
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
    g = padEnd(g, n)
    G = np.empty([k, n])
    for i in range(0, k):
        G[i,:] = shift(g, i)

    # row additions to gain embedded identity matrix on right side
    # -> systematic form
    if systematic:
        G = makeSystematic(G, verbose)

    return G.astype(int)

def makeSystematic(G, verbose = True):
    k, n = G.shape
    if verbose:
        print('unsystematic:');
        print(G.astype(int));
        print()

    # start with bottom row
    for i in range(k-1, 0, -1):
        if verbose: s = ''
        # start with most right hand bit
        for j in range(n-1, n-k-1, -1):
            # eleminate bit if it does not belong to identity matrix
            if G[i,j] == 1 and i != j-(n-k):
                if verbose: s += ' + g' + str(k-n+j)
                G[i,:] = (G[i,:] + G[k-n+j,:]) % 2

        if verbose and s != '':
            print('g' + str(i) + ' = g' + str(i) + s)
            print(G.astype(int));
            print()
    return G.astype(int)

def printAllCyclicCodes(factorPolynomials):
    """Generates all cyclic codes that can be created from
    the given factor polynomials.
    (slide 28)
    Args:
        factorPolynomials: factor polynomials in a python array
    """
    s = ''
    product = np.array([])
    for i in range(0, len(factorPolynomials)):
        if i == 0:
            product = factorPolynomials[i]
        else:
            product = GF2.multPoly(product, factorPolynomials[i])
        s += '(' + GF2.polyToString(factorPolynomials[i]) + ') '
    print(s + '= ' + GF2.polyToString(product))
    print()

    numberCodes = 2**(len(factorPolynomials)) - 2
    n = degree(product)
    print('There are', numberCodes, 'different cyclic codes of length', n, 'as')
    print('we can find', numberCodes, 'different generator polynomials that are')
    print('the factors of', GF2.polyToString(product))
    print(np.bitwise_and(1, 3))

    print('Code <- Generator polynomial')
    for i in range(0, numberCodes):
        s = ''
        gp = np.array([]) # generator polynomial
        for j in range(0, len(factorPolynomials)):
            if np.bitwise_and(i+1, 2**j) > 0:
                if s =='':
                    gp = factorPolynomials[j]
                else:
                    gp = GF2.multPoly(gp, factorPolynomials[j])
                s += '(' + GF2.polyToString(factorPolynomials[j]) + ')'

        print('Ccyc(' + str(n) + ', ' + str(degree(gp)) + ') <- g' + str(i+1) + ' = ' + s + ' = ' + GF2.polyToString(gp))

def padEnd(p, length):
    assert p.size <= length, \
        "padEnd() failed because polynomial is longer than given size."

    p = np.pad(p, (0, length-p.size), 'constant', constant_values=0)
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
        assert g[0] == 1, \
            "g0 must equal to 1"
        assert n >= degree(g), \
            "n=%i must be >= degree(g)=%i" % (n, degree(g))
        self._g = g[:n]; #auto remove too much dangling zeros
        self._n = n;

    def g(self):
        return self._g.astype(int)

    def printg(self):
        print(GF2.polyToString(self.g()))

    def n(self): # override
        return self._n

    def k(self): # override
        return self.n() - degree(self.g())

    def dmin(self, verbose = False): # override (LinearBlockCode dmin would work, but is slower)
        dmin = lbc.w(self.g())
        if verbose:
            print()
            print('Minimum Hamming distance (d_min) equals weight of generator polynomial g(X):')
            print('g(X) =', GF2.polyToString(self.g()))
            print('d_min =', dmin)
            print()
        return dmin

    def dminVerbose(self):
        self.dmin(True)

    def G(self, systematic = True, verbose = False): # override
        return gToG(self.g(), self.n(), systematic, verbose)

    def setG(self): # override
        assert False, "setG() not usable with cyclic codes."

    def setH(self): # override
        assert False, "setH() not usable with cyclic codes."

    def shift(self, c, i = 1):
        """Cyclic right shift of c using division (slide 11)
        """
        Xi = X(i) # X^i polynomial
        XiCX = GF2.multPoly(Xi, c) # X^i * c(X) polynomial
        Xn1 = GF2.addPoly(X(self.n()), X(0)) # X^n + 1 polynomial
        ci = GF2.modPoly(XiCX, Xn1) # i times shifted c
        return padEnd(ci, self.n())

    def c(self, m, systematic = True): # override
        """encode message polynomial m
        Args:
            m: message polynomial
            systematic: return codeword in systematic form (default: True)
        Returns:
            codeword
        """
        c = encode(m, self.g(), systematic)
        return padEnd(c, self.n())

    def printMessageCodewordTable(self, systematic = True): # override
        """Print all messages and their corresponding codewords.
        Args:
            systematic: print codewords in systematic form (default: True)
        """
        M = self.M()
        print('Messages -> Codewords')
        for m in M:
            c = self.c(m, systematic)
            print(m, c, 'm(X) =', GF2.polyToString(m), '\tc(X) =', GF2.polyToString(c) )

    def S(self, r):
        """Calculate Syndrome polynomial from receive or error polynomial.
        Args:
            r: receive or error polynomial
        Returns:
            Syndrome polynomial
        """
        return GF2.modPoly(r, self.g())

    def shiftSyndrome(self, S, i = 1):
        """Shift syndrome i times (slide 35)
        """
        for i in range(0, i):
            # S1(X) = XS(X) mod g(X)
            S = GF2.modPoly(GF2.multPoly(X(1), S), self.g())
        return S
