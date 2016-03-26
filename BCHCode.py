# -*- coding: UTF-8 -*-
import numpy as np
from CyclicCode import CyclicCode
from GaloisField import X, degree
import math

def g(GF, t, verbose = True):
    """Construct generator polynomial of BCH code with t.
    (slide 8)
    Args:
        GF: ExtendedGaloisField the roots are taken from
        t: number of correctable errors
        verbose: printout how the generator polynomial is generated
    Returns:
        Generator polynomial of BCH code with t
    """
    i_list = list(range(1, (2*t)+1, 2)) # only odd roots 1 <= i <= 2t

    if verbose:
        print("")
        print('Generator polynomial g(X) of BCH code with t = ' + str(t) + ':')
        print("")
        print('Assuming \u03B1 is a primitive element in GF(2^' + str(GF.m()) + '), ')
        print('the generator polynomial g(X) has the odd roots: \u03B1^i, i = 1, 3, ..., 2t')
        print('Assuming \u03A6i(X) is the minimal polynomial of \u03B1^i, then')
        print('g(X) can be expressed by the lowest common multiple (LCM) of the')
        print('minimal polynomials:')
        print("")
        # g(X) = LCM{ϕ_1(X), ϕ_2(X), . . . , ϕ_2t(X)}
        s = 'g(X) = LCM{'
        for i in i_list:
            s += '\u03A6_' + str(i) + '(X), '
        s = s[:-2] + '}'
        print(s)

    g = np.ones(1)
    for i in i_list:
        root = GF.elementFromExp(i)
        phi = GF.minimalPolynomial(GF.conjugateRoots(root))
        g = GF.multPoly(g, phi)
        if verbose:
            print('\u03A6_' + str(i) + '(X) = ' + GF.polyToString(phi))

    if verbose:
        print('g(X) = ' + GF.polyToString(g))
        print()

    return g

def HCF(A, B):
    """Calculate the highest common factor (HCF) of two integer numbers
    using the Euclidean Algorithm.
    (slide 24)
    """
    if A < B: # A has to be >= B
        tmp = A
        A = B
        B = tmp

    i = -1
    while True:
        # init values for i = -1 and i = 0
        if i == -1:
            ri = A
            si = 1
            ti = 0
            qi = '-'
        elif i == 0:
            ri = B
            si = 0
            ti = 1
            qi = '-'
        else:
        # recursive calculations
            qi = math.floor(ri_minus2 / ri_minus1)
            ri = ri_minus2 - qi * ri_minus1 # = ri_minus2 % ri_minus1
            si = si_minus2 - qi * si_minus1
            ti = ti_minus2 - qi * ti_minus1

        # break condition?
        if ri == 0:
            print(ri_minus1)
            return ri_minus1 # -> the previus r was the HCF

        # optional print
        print(i, '\t', ri, '\t', qi, '\t', si, '\t', ti)

        # store previous two values
        if i >= 0:
            ri_minus2 = ri_minus1
            si_minus2 = si_minus1
            ti_minus2 = ti_minus1
        if i >= -1:
            ri_minus1 = ri
            si_minus1 = si
            ti_minus1 = ti

        # increase i
        i += 1

class BCHCode(CyclicCode):
    """BCH Code

    Based on the the BCH Codes lecture (2016-25-02).

    Attributes:
        _GF: Galois Field the generator polynomial has roots in.
    """
    _GF = None

    def __init__(self, GF, t, verbose = False):
        """
        Args:
            GF: Galois Field the generator polynomial has roots in.
            t: Number of errors to correct.
            verbose: Print how the generator polynomial is constructed.
        """
        m = GF.m()
        self._GF = GF
        self._g = g(GF, t, verbose)
        self._n = 2**m - 1 # slide 5

    def GF(self):
        """ Return Galois Field.
        """
        return self._GF

    def H(self, systematic = True):
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
        for ni in range(0, n): # 0,1,2...n-1
            Hcol = None # ni'th column vector of H
            for ti in range(1, 2*t, 2): # 1,3,5...2t-1
                alpha = self.GF().elementFromExp(ti*ni) # (a^ti)^ni
                alpha = alpha[np.newaxis].T # row -> column vector
                if Hcol is None:
                    Hcol = alpha
                else:
                    Hcol = np.concatenate((Hcol, alpha), axis=0) # below
            if H is None:
                H = Hcol
            else:
                H = np.concatenate((H, Hcol), axis=1) # right
        return H.astype(int)

    def S(self, r, verbose = True): # override
        """Calculate Syndrome polynomial from receive or error polynomial.
        (slide 17, 26)
        Args:
            r: receive or error polynomial
        Returns:
            Syndrome polynomial
        """
        if verbose:
            print()
            print('The syndrome vector components are:')

        t = self.t()
        GF = self.GF()
        S = np.ones(2*t)

        for i in range(1, 2*t+1): # 1 <= i <= 2t
            exp_a = i
            a = GF.elementFromExp(exp_a)
            s = GF.substituteElementIntoPoly(r, a)
            S[i-1] = s

            if verbose:
                syndromeVectorStr = 's_' + str(i) + ' = r(' + GF.elementToString(a) + ') = '
                syndromeVectorStr += GF.polyToString(r, '(' + GF.elementToString(a) + ')') + ' = '
                syndromeVectorStr += GF.elementToString(s)
                print(syndromeVectorStr)

        if verbose:
            print()
            print('Therefore the syndrome polynomial is:')
            print('S(X) =', GF.polyToString(S))
            print()

        return S.astype(int)

    def decode(self, r, verbose = True):
        """Decode received polynomial r(X) using the Euclidean Algorithm.
        (slide 26)
        Args:
            r: received polynomial
            verbose: print the algorithm steps.
        """
        GF = self.GF()

        if verbose:
            print()
            print('Decode the received polynomial:')
            print('r(X) = ' + self.GF().polyToString(r))

        S = self.S(r, verbose)

        if verbose:
            print('The Euclidean algorithm is applied by constructing the ' + \
                  'following table:')

        ri, ti = GF.HCF(X(2 * self.t()), S, verbose)

        lamb = GF.monicMultiplier(ti)

        errorLocationPoly = GF.multPoly(lamb, ti);
        errorLocationPolyDerivative = GF.derivePoly(errorLocationPoly)
        errorEvalutationPoly = GF.multPoly(lamb, ri);

        if verbose:
            print(u'An element \u03BB \u2208 GF(' + str(GF.q()) + ') is conveniently selected to multiply')
            print(u't_i(X) by, in order to convert it into a monic polynomial.')
            print(u'This value of \u03BB is \u03BB = ' + GF.elementToString(lamb))
            print('Therefore:')
            print()
            print('Error location polynomial:')
            print(u'\u03C3(X) = \u03BB * t_i(X) = ' + GF.elementToString(lamb) + '(' + GF.polyToString(ti) + ')')
            print('                  = ' + GF.polyToString(errorLocationPoly) )
            print()
            print(u'\u03C3\'(X) = ' + GF.polyToString(errorLocationPolyDerivative))
            print()
            print('Error evaluation polynomial:')
            print(u'W(X) = -\u03BB * r_i(X) = ' + GF.elementToString(lamb) + ' * ' + GF.polyToString(ri))
            print('                   = ' + GF.polyToString(errorEvalutationPoly) )
            print()
            print(u'Performing Chien search in the error location polynomial \u03C3(X):')
            print()

        errorLocations = []
        for i, root in enumerate(GF.roots(errorLocationPoly)):
            j = GF.elementToExp(GF.elementFromExp(-GF.elementToExp(root)))
            errorLocations.append(j)
            if verbose:
                print(u'\u03B1^(-j_' + str(i+1) + ') = ' + GF.elementToString(root) + \
                     '\t-> j_' + str(i+1) + ' = ' + str(j))

        if verbose:
            print('\nError values:')

        errorValues = []

        for i, errorLocation in enumerate(errorLocations):
            alpha = GF.elementFromExp(-errorLocation)
            res_W = GF.substituteElementIntoPoly(errorEvalutationPoly, alpha)
            res_o = GF.substituteElementIntoPoly(errorLocationPolyDerivative, alpha)
            errorValue = GF.divElements(res_W, res_o)
            errorValues.append(errorValue)
            if verbose:
                print('e_j' + str(i+1) + ' = W(' + GF.elementToString(alpha) + \
                    u') / \u03C3\'(' + GF.elementToString(alpha) + ') = ' + \
                    GF.elementToString(res_W) + \
                    ' / ' + GF.elementToString(res_o) + ' = ' + \
                    GF.elementToString(errorValue))

        e = np.zeros(degree(r)+1)
        for i, errorLocation in enumerate(errorLocations):
            errorValue = errorValues[i]
            e[errorLocation] = errorValue

        c = GF.addPoly(r, e)

        if verbose:
            print()
            print('Error polynomial:')
            print('e(X) = ' + GF.polyToString(e))
            print()
            print('Code vector:')
            print('c = r + e = ' + str(c))

        return c.astype(int)


    def printInfo(self):
        """
        (slide 5, 7)
        """
        GF = self.GF()
        print()
        print('Generator Polynomial:        g(X) = ' + GF.polyToString(self.g()))
        rootStr = ''
        roots = GF.removeConjugateRoots(GF.roots(self.g()))
        for root in roots:
            rootStr += GF.elementToString(root) + ', '
        rootStr = rootStr[:-2] # remove last comma
        print('Roots of g(X) in GF(2^' + str(self.GF().m()) + '):    ' + rootStr + ' and all conjugate roots')
        print('Code length:                 n = 2^m - 1 = ' + str(self.n()))
        print('Message length:              k = ' + str(self.k()))
        print('Number of parity bits:       n - k = ' + str(self.n() - self.k()) + ' <= mt')
        print('Minimum Hamming distance:    dmin = ' + str(self.dmin()) + ' >= 2t + 1')
        print('Error correction capability: t = ' + str(self.t()))
        print()
