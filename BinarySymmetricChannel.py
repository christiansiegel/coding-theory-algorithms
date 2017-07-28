# -*- coding: UTF-8 -*-

import numpy as np
import math

class BSC:
    __P = np.zeros([2, 2], float)
    __PX = np.zeros(2, float)

    def __init__(self, PX, P):
        self.__P = np.array(P).astype(float)
        self.__PX = np.array(PX).astype(float)

    def PX(self, verbose = False):
        PX = self.__PX
        if verbose:
            for i in range(0, PX.size):
                print('P(X = ' + str(i) + ') = ' + str(PX[i]))
        if verbose: print('')
        return PX

    def PY(self, verbose = False):
        PYX = self.PYX()
        PX = self.PX()

        if verbose:
            print(u'P(yj) = \u2211i P(yj | xi) * P(xi)')

        PY = []
        for j in range(0, PYX.shape[0]):
            sumi = 0
            for i in range(0, PYX.shape[1]):
                sumi = sumi + PYX[j,i] * PX[i]
            if verbose:
                print('P(Y = ' + str(j) + ') = ' + str(sumi))
            PY.append(sumi)
        if verbose: print('')
        return np.array(PY).astype(float)

    def PXY(self, verbose = False):
        # Bayes theorem
        PYX = self.PYX()
        PX = self.PX()
        PY = self.PY()

        PXY = np.zeros(np.transpose(PYX).shape)

        for j in range(0, PYX.shape[0]):
            for i in range(0, PYX.shape[1]):
                PXY[i,j] = PYX[j,i] * PX[i] / PY[j]
                if(verbose):
                    print('P(X = ' + str(i) + ' | Y = ' + str(j) + ') = P(Y = ' + str(j) + ' | X = ' + str(i) + ') * P(X = ' + str(i) + ' ) /  P(Y = ' + str(j) + ') = ' + str(PXY[i,j]))
        if verbose: print('')
        return PXY

    def PYX(self, verbose = False):
        P = self.__P

        if verbose:
            for i in range(0, P.shape[0]):
                for j in range(0, P.shape[1]):
                    print('P(Y = ' + str(j) + ' | X = ' + str(i) + ') = ' + str(P[i,j]))
        if verbose: print('')
        return np.transpose(P)

    def HX(self, verbose = False):
        PX = self.PX()

        printStr = u'H(X) = \u2211i P(xi) * log2 (1/P(xi)) = '

        H = 0
        for i in range(0, PX.size):
            printStr = printStr + str(PX[i]) + ' * log2(1/' + str(PX[i]) + ') + '
            H = H + PX[i] * np.log2(1/PX[i])
        printStr = printStr[:-2]
        printStr = printStr + '= ' + str(H)

        if verbose:
            print(printStr)
            print('')
        return H

    def HY(self, verbose = False):
        PY = self.PY()

        printStr = u'H(Y) = \u2211j P(yj) * log2 (1/P(yj)) = '

        H = 0
        for j in range(0, PY.size):
            printStr = printStr + str(PY[j]) + ' * log2(1/' + str(PY[j]) + ') + '
            H = H + PY[j] * np.log2(1/PY[j])
        printStr = printStr[:-2]
        printStr = printStr + '= ' + str(H)

        if verbose:
            print(printStr)
            print('')
        return H

    def HXY(self, verbose = False):
        PXY = self.PXY()
        PY = self.PY()

        HXY = 0
        for j in range(0, PXY.shape[1]):
            for i in range(0, PXY.shape[0]):
                if PXY[i,j] == 0: continue
                HXY = HXY + PXY[i,j] * PY[j] * np.log2(1 / PXY[i,j])

        if(verbose):
            print(u'H(X|Y) = \u2211i P(xi|yj) * P(yj) * log2[1/P(xi|yj)] = ' + str(HXY))
            print('')
        return HXY

    def HYX(self, verbose = False):
        PYX = self.PYX()
        PX = self.PX()

        HYX = 0
        for j in range(0, PYX.shape[0]):
            for i in range(0, PYX.shape[1]):
                if PYX[j,i] == 0: continue
                HYX = HYX + PYX[j,i] * PX[i] * np.log2(1 / PYX[j,i])

        if(verbose):
            print(u'H(Y|X) = \u2211i P(yj|xi) * P(xi) * log2[1/P(yj|xi)] = ' + str(HYX))
            print('')
        return HYX

    def Cs(self, verbose = True):
        PX = self.__PX
        P = 1 / PX.size
        if(verbose):
            print('Since the noise entropy H(Y|X) is independent of the source probabilities,')
            print('then the channel capacity will be achieved when ')
            for i in range(0, PX.size):
                print('P(X='+str(i)+') = ' + str(P))
            print('\nThen:')

        backupPX = self.__PX
        self.__PX = self.__PX * 0 + P # manipulate PX temporarely

        PY = self.PY(True)

        if(verbose):
            print('Hence, there is Hmax(Y) = ')
        HmaxY = self.HY(True)
        HYX = self.HYX()
        Cs = HmaxY - HYX
        if verbose:
            print('Cs = Imax(X,Y) = Hmax(Y) - H(Y|X) = ' + str(HmaxY) + ' - ' + str(HYX) + ' = ' + str(Cs))
            print('')

        self.__PX = backupPX # restore

        return Cs

    def ChannelEfficiency(self, verbose = True):
        Cs = self.Cs(False)
        IXY = bsc.HY() - bsc.HYX()
        eta = IXY / Cs
        if(verbose):
            print(u'\u03B7 = I(X,Y) / Cs = ' + str(IXY) + ' / ' + str(Cs) + ' = ' + str(eta))
            print('')
        return eta


if __name__ == '__main__':

    """
    # Exam 2011 - 1
    PX = [0.2, 0.8]
    P = [[0.8, 0.1, 0.1],
         [0.1, 0.1, 0.8]]

    bsc = BSC(PX, P)
    bsc.PX(True)
    bsc.HX(True)
    bsc.HY(True)
    bsc.PYX(True)
    bsc.PY(True)
    bsc.PXY(True)
    bsc.HXY(True)
    bsc.HYX(True)

    print('I(X,Y) = H(X) - H(X|Y) = ' + str(bsc.HX() - bsc.HXY()))
    print('I(X,Y) = H(Y) - H(Y|X) = ' + str(bsc.HY() - bsc.HYX()))

    bsc.Cs(True)
    bsc.ChannelEfficiency()"""

    # Exam 2015 - 1.4
    PX = [1/3, 1/3, 1/3]
    P = [[1/3, 1/3, 1/3, 0, 0],
         [0, 1/3, 1/3, 1/3, 0],
         [0, 0, 1/3, 1/3, 1/3]]
    bsc = BSC(PX, P)
    bsc.HXY(True)
    bsc.Cs()
