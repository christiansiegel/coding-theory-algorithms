from LinearBlockCode import LinearBlockCode
import numpy as np
import GaloisFields as gf
import LinearBlockCode as lbc

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

def polyToString(p, variable = 'X'):
    """Returns polynomial in string representation.
    E.g. [0,1,0,3] -> "X + 3*X^3"

    Args:
        v: polynomial
        variable: variable used in string
    Returns:
        Polynomial p in string representation.
    """
    s = ''
    for i in range(0, p.size):
        if p[i] > 0:
            if s != '':
                s += ' + '
            if i == 0:
                s += str(p[i])
            elif i == 1:
                if p[i] > 1:
                    s += str(p[i]) + '*'
                s += variable
            else:
                if p[i] > 1:
                    s += str(p[i]) + '*'
                s += variable + '^' + str(i)
    if s == '':
        s = '0'
    return s

def addPoly(a, b):
    """Add two polynomials; GF(2)
    (slide 7)
    """
    pa = np.poly1d(np.flipud(a))
    pb = np.poly1d(np.flipud(b))
    return np.flipud(np.poly1d(pa + pb).c % 2)

def multPoly(a, b):
    """Multiply two polynomials; GF(2)
    (slide 7)
    """
    pa = np.poly1d(np.flipud(a))
    pb = np.poly1d(np.flipud(b))
    return np.flipud(np.poly1d(pa * pb).c % 2)

def divPoly(a, b):
    """Divide two polynomials; GF(2) -> returns quotient
    (slide 8)
    """
    pa = np.poly1d(np.flipud(a))
    pb = np.poly1d(np.flipud(b))
    divisionResults = pa / pb
    q = divisionResults[0] # quotient
    return np.flipud(q.c % 2).astype(int)

def modPoly(a, b):
    """Divide two polynomials; GF(2) -> returns remainder
    (slide 8)
    """
    pa = np.poly1d(np.flipud(a))
    pb = np.poly1d(np.flipud(b))
    divisionResults = pa / pb
    r = divisionResults[1] # remainder
    return np.flipud(r.c % 2).astype(int)

def X(i):
    """Create single coefficient polynomial with degree i: X^i
    """
    X = np.zeros(i+1)
    X[i] = 1
    return X.astype(int)

def encode(m, g, systematic = True):
    """Encoding of cyclic code (in systematic form)
    (slide 23)
    ATTENTION: Dangling zeros in returned codeword are cut away.
    """
    if systematic:
        r = gf.degree(g) # r = n - k
        Xr = X(r) # X^(n-k)
        XrmX = multPoly(Xr, m) # X^(n-k) * m(X)
        p = modPoly(XrmX, g) # p(X) = (X^(n-k) * m(X)) mod g(X)
        c = addPoly(p, XrmX) # c(X) = p(X) + (X^(n-k) * m(X))
    else:
        c = multPoly(m, g)
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
    k = n - gf.degree(g)
    g = padEnd(g, n)
    G = np.empty([k, n])
    for i in range(0, k):
        G[i,:] = shift(g, i)

    # row additions to gain embedded identity matrix on right side
    # -> systematic form
    if systematic:
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
            product = multPoly(product, factorPolynomials[i])
        s += '(' + polyToString(factorPolynomials[i]) + ') '
    print(s + '= ' + polyToString(product))
    print()

    numberCodes = 2**(len(factorPolynomials)) - 2
    n = gf.degree(product)
    print('There are', numberCodes, 'different cyclic codes of length', n, 'as')
    print('we can find', numberCodes, 'different generator polynomials that are')
    print('the factors of', polyToString(product))
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
                    gp = multPoly(gp, factorPolynomials[j])
                s += '(' + polyToString(factorPolynomials[j]) + ')'

        print('Ccyc(' + str(n) + ', ' + str(gf.degree(gp)) + ') <- g' + str(i+1) + ' = ' + s + ' = ' + polyToString(gp))

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
        assert n >= gf.degree(g), \
            "n=%i must be >= degree(g)=%i" % (n, gf.degree(g))
        self._g = g[:n]; #auto remove too much dangling zeros
        self._n = n;

    def g(self):
        return self._g.astype(int)

    def printg(self):
        print(polyToString(self.g()))

    def n(self): # override
        return self._n

    def k(self): # override
        return self.n() - gf.degree(self.g())

    def dmin(self): # override (LinearBlockCode dmin would work, but is slower)
        return lbc.w(self.g())

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
        XiCX = multPoly(Xi, c) # X^i * c(X) polynomial
        Xn1 = addPoly(X(self.n()), X(0)) # X^n + 1 polynomial
        ci = modPoly(XiCX, Xn1) # i times shifted c
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
            print(m, c, 'm(X) =', polyToString(m), '\tc(X) =', polyToString(c) )

    def S(self, r):
        """Calculate Syndrome polynomial from receive or error polynomial.
        Args:
            r: receive or error polynomial
        Returns:
            Syndrome polynomial
        """
        return modPoly(r, self.g())

    def shiftSyndrome(self, S, i = 1):
        """Shift syndrome i times (slide 35)
        """
        for i in range(0, i):
            # S1(X) = XS(X) mod g(X)
            S = modPoly(multPoly(X(1), S), self.g())
        return S
