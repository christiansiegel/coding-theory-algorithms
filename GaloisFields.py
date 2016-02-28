import numpy as np
import CyclicCode as cc
import LinearBlockCode as lbc

"""
TODO: Maybe add mult, add, div and mod function (slide 4)
      and remove them in CyclicCode.py.
"""

def degree(p):
    """Returns degree of polynomial (highest exponent).
    (slide 3)
    """
    poly = np.poly1d(np.flipud(p))
    return poly.order

def constructGF2(p, verbose = True):
    """Construct GF(2^m) based on primitive polynomial p.
    The degree of pi(X) is used to determine m.
    (slide 12)
    Args:
        p: primitive polynomial p to construct the GF with.
        verbose: print information on how the GF is constructed.
    Returns:
        Elements of the GF in polynomial representation.
    """
    elements = []
    m = degree(p)
    a_high = p[:m] # see slide 12 Solution

    if verbose:
        print()
        print('Construct a GF(2^' + str(m) + ') based on primitive')
        print('polynomial pi(X) =', cc.polyToString(p, 'X'))
        print()
        print('Assuming pi(X) has root \u03B1, there is')
        print('pi(\u03B1) =', cc.polyToString(p, '\u03B1'), '= 0,')
        print('then \u03B1^' + str(m) + ' = ' + cc.polyToString(a_high, '\u03B1'))
        print()
        print('Exp. rep.\t vector rep.\t poly. rep.')
        print('-------------------------------------------')

    for i in range(0, 2**m):
        # create exponential representation
        if i == 0:
            exp = np.array([0])
        else:
            exp = cc.X(i-1)

        # create polynomial representation (CAN'T EXPLAIN...IT'S MAGIC)
        poly = exp
        if degree(poly) >= m:
            quotient, remainder = divmod(degree(poly), m)

            poly = cc.X(remainder)
            for j in range(0, quotient):
                poly = cc.multPoly(poly, a_high)

            while degree(poly) >= m:
                poly = cc.addPoly(poly, elements[degree(poly) + 1])
                poly = poly[:-1]

        # format polynomial (size m)
        poly = poly[:degree(poly) + 1]
        poly = cc.padEnd(poly, m)

        # append to elements list for return
        elements.append(poly.astype(int))

        # print row
        if verbose:
            expStr = cc.polyToString(exp, '\u03B1')
            polyStr = cc.polyToString(poly, '\u03B1')
            print(expStr, '\t\t', poly, '\t', polyStr)

    if verbose: print()
    return elements

def irreducible(p, verbose = False):
    """
    Test all factor polynomials over GF(2) of degree higher than
    zero and lower than m to see if p has no factor polynomial and
    thus is irreducible ofer GF(2).
    (slide 6)
    Args:
        p: polynomial to check if irreducible over GF(2)
    Returns:
        True if polynomial is irreducible.
    """
    m = degree(p)
    irreducible = True
    if verbose:
        print()
        s = ''
    for factorPoly in allPolynomialsWithDegreeBetween(0, m):
        if isFactor(p, factorPoly):
            if verbose:
                irreducible = False
                s += '\n' + cc.polyToString(factorPoly)
            else:
                return False # skip all other tests
    if verbose:
        if irreducible:
            print('The polynomial', cc.polyToString(p), \
                  'is irreducible over GF(2), since \nit has no', \
                  'factor polynomials over GF(2) of degree higher than\n'+ \
                  'zero and lower than ' + str(m) + '.')
        else:
            print('The polynomial', cc.polyToString(p), \
                  'is NOT irreducible over GF(2).\n' + \
                  'It has the following factor polynomials:' + s)
    if verbose: print()
    return irreducible

def primitive(p, verbose = False):
    """
    Test of polynomial is primitive (and hence also irreducible).
    (slide 7)
    Args:
        p: polynomial to check if primitive
    Returns:
        True if polynomial is primitive.
    """
    if not irreducible(p, verbose):
        if verbose:
            print('Hence, the polynomial is also not primitive.')
        return False
    else: # irreducible
        m = degree(p)
        for n in range(1, 2**m-1):
            p2 = cc.addPoly(cc.X(0), cc.X(n))
            if isFactor(p2, p):
                if verbose:
                    print('The polynomial', cc.polyToString(p), \
                          'is a factor polynomial of 1+X^' + str(n) + '\n' + \
                          'and hence not primitve.')
                return False
    if verbose:
        print('The polynomial', cc.polyToString(p), 'is also primitve,',\
              'since it is not a \nfactor of 1+X^n, 1 <= n < ' + \
              str(2**m-1) + '.\n')
    return True

def isFactor(p, factorPoly):
    """Check if polynomial factorPoly is a factor of polynomial p.
    """
    remainder = cc.modPoly(p, factorPoly)
    q = cc.divPoly(p, factorPoly)
    return np.count_nonzero(remainder) == 0

def allPolynomialsWithDegreeBetween(lower, upper):
    """Generate all polynomials with lower < degree(p) < upper.
    Args:
        lower: lower degree bound (exclusive)
        upper: upper degree bound (exclusive)
    Returns:
        All polynomials within given degree bounds as standard
        python array.
    """
    polynomials = []
    for i in range(1, 2**(upper)): # ensures factor polys with degree < m
        p = lbc.intToArray(i)
        if degree(p) > lower: # ensures factor polys with degree > 0
            polynomials.append(p)
    return polynomials

class ExtendedGaloisField:
    """Extended Galois Field GF(2^m)

    Based on the the Galois Field lecture (2016-18-02).

    Attributes:
        _p: Primitive polynomial pi(X) the GF is based on.
        _cachedElements: All elements of the GF in polynomial representation
    """
    _p = np.zeros(1)
    _cachedElements = []

    def __init__(self, p):
        """Create Galois Field GF(2^m)
        The degree of pi(X) is used to determine m.

        Args:
            p: Primitive polynomial pi(X) the GF is based on.
        """
        self._p = p
        self._cachedElements = constructGF2(self.p(), False)

    def p(self):
        """Primitive polynomial pi(X) the GF is based on.
        """
        return self._p.astype(int)

    def m(self):
        """GF(2^m) -> returns m
        """
        return degree(self.p())

    def printInfo(self):
        """Prints how the GF is constructed from the primitive
        polynomial pi(X).
        """
        constructGF2(self.p(), True)

        """ slide 17: """
        m = self.m()
        tmp = cc.addPoly(cc.X(0), cc.X(2**m-1))
        print('-> The non-zero elements of GF(2^' + str(self.m()) + \
              ') are all roots of ' + cc.polyToString(tmp) + '.')

        tmp = cc.addPoly(cc.X(1), cc.X(2**m))
        print('-> The elements of GF(2^' + str(self.m()) + \
              ') are all roots of ' + cc.polyToString(tmp) + '.')
        print()

    def numberOfNonZeroElements(self):
        """Number of non-zero elements in the GF.
        """
        return len(self._cachedElements) - 1

    def getAllElements(self):
        """All elements of the GF in polynomial representation.
        """
        return self._cachedElements

    def resolveElementIndex(self, i):
        """E.g. GF(2^m) has 2**m-1=15 non-zero elements.
        alpha^16 is the same as alpha^1.
        If parameter i=16 this function will return 1.
        """
        number = self.numberOfNonZeroElements()
        if i >= number:
            i = i % number
        return i

    def getElement(self, i):
        """Element alpha^i in polynomial representation.
        For negative exponents the zero element is returned.
        """
        if i < 0:
            return np.zeros(1).astype(int)
        i = self.resolveElementIndex(i)
        return self._cachedElements[i + 1]

    def exponentOfElement(self, alpha):
        """Find element alpha^i in GF an return its exponent i.
        If element is the zero element or the element is not found the
        exponent -1 is returned.
        """
        for index, item in enumerate(self.getAllElements()):
            item = item[:degree(item)+1]
            alpha = alpha[:degree(alpha)+1]
            if np.array_equal(item, alpha):
                return index - 1
        return -1

    def subsituteAlpha(self, p, i):
        """Subsitutes alpha^i in given polynomial and returns result.
        Args:
            p: polynomial alpha should be subsituted in
            i: exponent of alpha to be subsituted (alpha^i)
        """
        result = np.zeros(degree(p))
        for j in range(0, degree(p) + 1): # iterate all coefficients
            if p[j] == 0:
                continue
            exponent = j * i # (alpha^i)^j = alpha^exponent
            alpha = self.getElement(exponent)
            result = cc.addPoly(result, alpha)
        return result.astype(int)

    def roots(self, p, allRoots = True):
        """Subsitutes all elements of the GF into polynomial p to
        find roots and returns them in a standard python array.
        (slide 14)
        Args:
            p: polynomial to find roots for in the GF.
        Returns:
            Roots of p in the GF as standard python array.
        """
        roots = []
        number = self.numberOfNonZeroElements()
        for i in range(0, number):
            palpha = self.subsituteAlpha(p, i)
            if np.count_nonzero(palpha) == 0:
                roots.append(i)
        if allRoots:
            return roots
        else:
            return self.removeConjugateRoots(roots)

    def conjugateRoots(self, i, verbose = False):
        """
        Conjugate a known root (beta = alpha^i) to all other roots
        (beta^(2^l)) of this conjugate root group and return them.
        (slide 16)
        Args:
            i: exponent of the known root (alpha^i)
            verbose: print info on how this is calculated
        Returns:
            Sorted conjugate roots group in the GF as standard
            python array.
        """
        if verbose: print()
        roots = []
        l = 0
        while True:
            l += 1
            real_i = self.resolveElementIndex(i*2**l)
            roots.append(real_i)
            if verbose:
                print('l = ' + str(l) + ':\t(\u03B1^' + str(i) + \
                      ')^' + str(2**l) + ' = \u03B1^' + str(i*2**l) + \
                      ' = \u03B1^' + str(real_i))

            if(real_i == i):
                break
        if verbose: print()
        return sorted(roots)

    def removeConjugateRoots(self, roots):
        """Remove all conjugate roots from given roots list and leave only
        lowest root.
        """
        result = roots
        for root in roots:
            conjugateRoots = self.conjugateRoots(root)
            for conjugateRoot in conjugateRoots[1:]: # don't remove first root
                if conjugateRoot in result:
                    result.remove(conjugateRoot)
        return result

    def conjugateRootGroups(self):
        """Calculate all conjugate groups and return them in a
        python array. E.g.
        Conjugate roots:  [0][1],[a, a^2, a^4, a^8],[a^3, a^6, a^9, a^12],...
        Returned array: [[-1][0],[0,2,4,8],[3,6,9,12],...]

        ATTENTION: Conjugate root group [0] is denoted as negative exponent.
        """
        groups = [[-1]]
        m = self.m()
        for i in range(0, 2**m-1):
            group = self.conjugateRoots(i)
            if group not in groups:
                groups.append(group)
        return groups

    def printMinimalPolynomials(self):
        """Print all conjugate root groups and their corresponding
        minimal polynomial.
        """
        print()
        print('Conjugate roots  ->  Minimal polynomials')
        print('----------------------------------------')
        for rootGroup in self.conjugateRootGroups():
            rootStr = ''
            for root in rootGroup:
                if root == -1: # special case: 0 is denoted as negative exponent
                    rootStr += '0'
                else:
                    rootStr += '\u03B1^' + str(root) + ', '
            rootStr = rootStr[:-2] # remove last comma
            minPoly = self.minimalPolynomial(rootGroup)
            print(rootStr, '\t', cc.polyToString(minPoly))
        print()

    def minimalPolynomial(self, conjugateRoots):
        """Generate minimal polynomial from conjugate root group.
        Args:
            conjugateRoots: Exponent of roots in a standard python array
                            The zero element is denoted with the exponent -1.
        """
        # Better not ask how this works...but it does.
        aexp = [-1]
        for root in conjugateRoots:
            aexpNew = [-1] * (len(aexp) + 1)
            for i in range(0, len(aexp)):
                rootExp = [root, -1]
                for j in range(0, len(rootExp)):
                    # add exponents (e.g. a^4 * a^5 = a^9)
                    # threat zero elements -1 exponent as if it isn't there
                    if aexp[i] >= 0 and rootExp[j] >= 0:
                        newExp = rootExp[j] + aexp[i]
                    elif rootExp[j] >= 0:
                        newExp = rootExp[j]
                    else:
                        newExp = aexp[i]
                    # binary addition of alphas in GF
                    newpoly = cc.addPoly(\
                        self.getElement(aexpNew[i+j]),\
                        self.getElement(newExp))
                    aexpNew[i+j] = self.exponentOfElement(newpoly)
            aexpNew[i+j] = 0 #
            aexp = aexpNew

        #convert internal weird format to polynomial
        polynomial = np.empty(len(aexp))
        for i in range(0, polynomial.size):
            polynomial[i] = aexp[i] + 1

        return polynomial.astype(int)

class GaloisField(ExtendedGaloisField):
    """Galois Field GF(2)
    """
    def __init__(self):
        self._p = np.array([1,1])
        self._cachedElements = constructGF2(self.p(), False)
