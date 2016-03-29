# -*- coding: UTF-8 -*-

from BinarySymmetricChannel import BinarySymmetricChannel
from GaloisField import GaloisField
from LinearBlockCode import LinearBlockCode
from CyclicCode import CyclicCode
from BCHCode import BCHCode
import numpy as np
import CyclicCode as cyccode
from GaloisField import GF2

def Exercise2_4():
    '''
    Exercise 2.4
    Codewords = np.array([[0,0,0,0,0,0],  0
                         [0,1,1,1,0,0], 3
                         [1,0,1,0,1,0], 3
                         [1,1,0,1,1,0], 4
                         [1,1,0,0,0,1], 3
                         [1,0,1,1,0,1], 4
                         [0,1,1,0,1,1], 4
                         [0,0,0,1,1,1]]) 3

    (a) What is the rate of the code?
    (b) Write down the generator and parity check matrices of this code in systematic form.
    (c) What is the minimum Hamming distance of the code?
    (d) How many errors can it correct, and how many can it detect?
    (e) Compute the syndrome vector for the received vector r = (101011) and hence find the location of any error.


    n = 6 and k = 3 because 2^k = 8
    so we need to choose 3 vectors that form an identity matrix
    Generator thus becomes
    '''
    print("Exercise 2.4")
    Generator = np.array([[0,1,1,1,0,0],
                         [1,0,1,0,1,0],
                         [1,1,0,0,0,1]])

    lbc = LinearBlockCode()
    lbc.setG(Generator)
    lbc.printInfo()
    print("")
    r = np.array([1,0,1,0,1,1])
    lbc.verboseSyndromeDecode(r)

def Exercise2_5():
    print("Exercise 2.5")
    '''
    (a) Construct a linear block code Cb(5, 2), maximizing its minimum Hamming distance.
    (b) Determine the generator and parity check matrices of this code.
    '''
    # Create generator matrix (right submatrix has dist 2 and the lh submatrix can only have 1 variation due to I)
    # Dmin has to be at least k + 1 if we are maximizing it.
    Generator = np.array([[1,1,1,1,0],
                           [0,1,1,0,1]])
    lbc = LinearBlockCode()
    lbc.setG(Generator)
    lbc.printInfo()

def Exercise2_6():
    print("Exercise 2.6")
    '''
        1 1 0 1 1 0 0 1 1 0 1 0 0
    G   1 0 1 1 0 1 0 1 0 1 0 1 0
        1 1 1 0 0 0 1 1 1 1 0 0 1
    (a) Find the parity check matrix H and hence write down the parity check equations.
    (b) Find the minimum Hamming distance of the code.
    '''
    Generator = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                          [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1]])
    lbc = LinearBlockCode()
    lbc.setG(Generator)
    lbc.printInfo()

def Exercise2_7():
    print("Exercise 2.7")
    '''
    The generator matrix of a binary linear block code is given below:
    G =    1 1 0 0 1 1 1 0
           0 0 1 1 1 1 0 1

    (a) Write down the parity check equations of the code.
    (b) Determine the code rate and minimum Hamming distance.
    (c) If the error rate at the input of the decoder is 10−3, estimate the error rate at
    the output of the decoder.
    '''
    Generator = np.array([[1, 1, 0, 0, 1, 1, 1, 0],[0, 0, 1, 1, 1, 1, 0, 1]])
    lbc = LinearBlockCode()
    lbc.setG(Generator)
    lbc.printInfo()
    # [([n-1][t]])*p^(t+1)
    #print("Error rate at output: " , np.array([[7],[2]])*(10^3)^3)

def Exercise2_8():
    print("Exercise 2.8")
    '''
    The Hamming block code Cb(15, 11) has the following parity check submatrix:
        ⎡0 0 1 1⎤
        ⎢0 1 0 1⎥
        ⎢1 0 0 1⎥
        ⎢0 1 1 0⎥
        ⎢1 0 1 0⎥
    P = ⎢1 1 0 0⎥
        ⎢0 1 1 1⎥
        ⎢1 1 1 0⎥
        ⎢1 1 0 1⎥
        ⎢1 0 1 1⎥
        ⎣1 1 1 1⎦

    (a) Construct the parity check matrix of the code.
    (b) Construct the error pattern syndrome table.
    (c) Apply syndrome decoding to the received vector r = (011111001011011).
    '''
    Generator = np.array([[0, 0, 1, 1],
                          [0, 1, 0, 1],
                          [1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [1, 0, 1, 0],
                          [1, 1, 0, 0],
                          [0, 1, 1, 1],
                          [1, 1, 1, 0],
                          [1, 1, 0, 1],
                          [1, 0, 1, 1],
                          [1, 1, 1, 1]]).transpose()
    lbc = LinearBlockCode()
    lbc.setG(Generator)
    lbc.printInfo()

    r = np.array([0,1,1,1,1,1,0,0,1,0,1,1,0,1,1])
    lbc.syndromeDecode(r)

def Exercise3_2():
    print("Exercise 3.2")
    '''
    Verify that the generator polynomial g(X) = 1 + X + X2 + X3 generates a binary
    cyclic code Ccyc(8, 5) and determine the code polynomial for the message vector
    m = (10101) in systematic form.
    '''
    g = np.array([1,1,1,1])
    cc = CyclicCode(g, 8)
    cc.printg()
    # cc.c(np.array([1,0,1,0,1]))
    # result [1 0 1 0 1] [0 1 0 1 0 1 0 1] c(X) = X + X^3 + X^5 + X^7
    cc.printInfo()

def Exercise3_3():
    print("Exercise 3.3")
    '''
    A binary linear cyclic code Ccyc(n, k) has code length n = 7 and generator polynomial g(X) = 1 + X2 + X3 + X4.
    (a) Find the code rate, the generator and parity check matrices of the code in systematic form, and its Hamming distance.
    (b) If all the information symbols are ‘1’s, what is the corresponding code vector?
    (c) Find the syndrome corresponding to an error in the first information symbol,
    and show that the code is capable of correcting this error.
    '''
    g = np.array([1,0,1,1,1])
    cc = CyclicCode(g, 7)
    cc.printInfo()

def Exercise3_6():
    print("Exercise 3.6")
    '''
    a) Determine the table of code vectors of the binary linear cyclic block code
    Ccyc(6, 2) generated by the polynomial g(X) = 1 + X + X3 + X4.
    (b) Calculate the minimum Hamming distance of the code, and its errorcorrection
    capability.
    '''

def exam2011problem2():
    lbc = LinearBlockCode()
    H = np.array([[1,0,0,0,0,1,1,1],
                  [0,1,0,0,1,1,1,0],
                  [0,0,1,0,1,1,0,1],
                  [0,0,0,1,1,0,1,1]])
    lbc.setH(H)
    lbc.printInfo()
    r = np.array([0,1,1,1,0,1,1,0])
    lbc.verboseSyndromeDecode(r)

def exam2011problem3():
    G = np.array([  [1,1,1,0,1,1,0,0,1,0,1,0,0,0,0],
                    [0,1,1,1,0,1,1,0,0,1,0,1,0,0,0],
                    [0,0,1,1,1,0,1,1,0,0,1,0,1,0,0],
                    [0,0,0,1,1,1,0,1,1,0,0,1,0,1,0],
                    [0,0,0,0,1,1,1,0,1,1,0,0,1,0,1]])

    g = np.array([1,1,1,0,1,1,0,0,1,0,1])
    X15 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    print(GF2.polyToString(GF2.divPoly(X15, g)))

    G = cyccode.makeSystematic(G, True)

    lbc = LinearBlockCode()
    lbc.setG(G)
    r = np.array([1,0,1,1,0])
    c = lbc.c(r)
    print(c)

def exam2011problem4():
    p = np.array([1,1,0,0,1])
    GF16 = GaloisField(p)
    GF16.printInfo()
    t = 2
    bch = BCHCode(GF16, t, True)
    bch.printInfo()
    r = np.array([0,0,0,1,1,1,1,1,1,1,0,0,0,1,1])
    bch.S(r, True)

    equation = np.array([1,1,1])
    root1, root2 = GF16.roots(equation)
    j1 = GF16.elementToExp(GF16.elementFromExp(-GF16.elementToExp(root1)))
    j2 = GF16.elementToExp(GF16.elementFromExp(-GF16.elementToExp(root2)))
    print(j1, j2)

    c = bch.decode(r, True)
    print(GF16.polyToString(c))

def exam2014problem2():
    #1
    H = np.array([[1,0,0,0,1,1],
                  [0,1,0,1,0,1],
                  [0,0,1,1,1,0]])
    lbc = LinearBlockCode()
    lbc.setH(H)
    print(lbc.G())
    #2
    Gdual = H
    print(Gdual)
    #3,4
    lbcDual = LinearBlockCode()
    lbcDual.setG(Gdual)
    lbcDual.printInfo()

def exam2014problem3():
    g1 = np.array([1,0,0,1])
    g2 = np.array([1,1,1,1])
    r, t = GF2.HCF(g1, g2, True)
    print(GF2.polyToString(t))

if __name__ == '__main__':
    print("")
    #g1 = np.array([1, 1, 1 ,1 ,0 ,0 ,1])
    #g2 = np.array([1 ,0 ,1, 1, 0 ,1 ,1])
    #r, t = GF2.HCF(g1, g2, True)
    #print(GF2.divPoly(g1, t))
    #Exercise2_4()
    #Exercise2_5()
    #Exercise2_6()
    #Exercise2_7()
    #Exercise2_8()
    #Exercise3_2()
    Exercise3_3()
    #exam2011problem4()


'''
    # LinearBlockCode example:
    G = np.array([[1,1,0,1,0,0,0],
                  [0,1,1,0,1,0,0],
                  [1,1,1,0,0,1,0],
                  [1,0,1,0,0,0,1]])

    lbc = LinearBlockCode()
    lbc.setG(G)
    lbc.printInfo()

    # CyclicCode example:
    G = np.array([[1,1,0,1,0,0,0],
                  [0,1,1,0,1,0,0],
                  [1,1,1,0,0,1,0],
                  [1,0,1,0,0,0,1]])
    g = np.array([1,1,0,1])     # 1 + X + X^3;
    cc = CyclicCode(g, 7)       # Ccyc(7,4)
    cc.printInfo()


    # GaloisFields example:
    ## GF(2^4) generated by pi(X) = 1 + X + X^4
    GF16 = GaloisField(np.array([1,1,0,0,1]))
    GF16.printInfo()

    ## find roots of p(X) = 1 + X^3 + X^4 in GF(2^4)
    p = np.array([1,0,0,1,1])
    print("roots of", GF16.polyToString(p), "in GF(2^4):")
    for root in GF16.roots(p):
        print(GF16.elementToString(root))

    # BCHCode example:
    pX = np.array([1,1,0,0,1]) #  1 + X + X^4
    GF24 = GaloisField(pX)
    t = 2
    C_BCH = BCHCode(GF24, t, True)
'''
