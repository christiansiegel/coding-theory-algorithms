from BinarySymmetricChannel import BinarySymmetricChannel
from LinearBlockCode import LinearBlockCode
from CyclicCode import CyclicCode
from GaloisField import GaloisField
from BCHCode import BCHCode
import numpy as np
import CyclicCode as cc
import BCHCode as bch


if __name__ == '__main__':

    # Exercise 4.2 (precondition for 4.3)
    pX = np.array([1,0,1,0,0,1]) # 1 + X^2 + X^5
    GF25 = GaloisField(pX)
    #GF25.printInfo()

    print("Exercise 4.3")
    print("------------")

    GF25.printMinimalPolynomials()

    print("Exercise 4.4")
    print("------------")

    print(GF25.roots(pX))
    t = 3
    C_BCH = BCHCode(GF25, t, True)

    print("Exercise 4.5")
    print("------------")

    t = 2
    C_BCH = BCHCode(GF25, t, True)
    C_BCH.printInfo()

    print("Exercise 4.6 (a)")
    print("----------------")

    poly1 = np.array([1,1])       # (1 + X)
    poly2 = np.array([1,1,0,0,1]) # (1 + X + X^4)
    poly3 = np.array([1,1,1,1,1]) # (1 + X + X^2 + X^3 + X^4)

    g = GF25.multPoly(GF25.multPoly(poly1, poly2), poly3)
    n = 15
    C_BCH = CyclicCode(g, n) # use CyclicCode because it allows to specify n
    C_BCH.dmin(True)

    print("Exercise 4.9")
    print("------------")

    pX = np.array([1,1,0,0,1]) #  1 + X + X^4
    GF24 = GaloisField(pX)
    #GF24.printInfo()

    t = 2
    C_BCH = BCHCode(GF24, t, True)

    r = np.array([1,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    C_BCH.decode(r, True)
