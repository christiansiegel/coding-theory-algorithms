# -*- coding: UTF-8 -*-
import numpy as np
from BCHCode import BCHCode
from GaloisField import X, degree
import math

def g(GF, t, verbose = True):
    """Construct generator polynomial of RS code with t.
    (p 116)
    Args:
        GF: ExtendedGaloisField
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
        print(u'g(X) = (X - \u03B1)(X - \u03B1^2)...(X - \u03B1^(n-k))')
        print("")

    g = np.ones(1)
    for i in range(1, 2*t+1): # n - k = 2*t
        root = GF.elementFromExp(i)
        phi = np.array([root,1])
        g = GF.multPoly(g, phi)
        if verbose:
            print('\u03A6_' + str(i) + '(X) = ' + GF.polyToString(phi))

    if verbose:
        print('g(X) = ' + GF.polyToString(g))
        print()

    return g


class RSCode(BCHCode):
    """RS Code

    Attributes:
        _GF: Galois Field the generator polynomial has roots in.
    """
    _GF = None
    _t = 0

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
        self._t = t

    def GF(self):
        """ Return Galois Field.
        """
        return self._GF



    def printInfo(self):

        GF = self.GF()
        t = self._t
        print()
        print('Generator Polynomial:        g(X) = ' + GF.polyToString(self.g()))
        rootStr = ''
        roots = GF.removeConjugateRoots(GF.roots(self.g()))
        for root in roots:
            rootStr += GF.elementToString(root) + ', '
        rootStr = rootStr[:-2] # remove last comma
        print('Roots of g(X) in GF(2^' + str(self.GF().m()) + '):    ' + rootStr + ' and all conjugate roots')
        print('Code length:                 n = q - 1 = ' + str(self.n()))
        print('Message length:              k = ' + str(self.k()))
        print('Number of parity bits:       n - k = 2t = ' + str(2*t))
        print('Minimum Hamming distance:    dmin = ' + str(2*t+1) + ' = 2t + 1')
        print('Error correction capability: t = ' + str(t))
        print()
