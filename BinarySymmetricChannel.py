# -*- coding: UTF-8 -*-

import numpy as np
import math

_CAPACITY_TOLERANCE_BITS = 1e-12
_CAPACITY_MAX_ITERATIONS = 100000

def _capacity_bounds(P, logP, positive, log_source):
    """Return a source and its mutual-information lower/dual upper bounds, in nats."""
    log_output = np.logaddexp.reduce(log_source[:, np.newaxis] + logP, axis=0)
    log_ratio = np.zeros(P.shape)
    np.subtract(logP, log_output, out=log_ratio, where=positive)
    divergence = np.sum(P * log_ratio, axis=1)
    source = np.exp(log_source)
    # A candidate may leave an output unreachable and give unused rows infinite
    # divergence. Its upper bound is then infinite; avoid zero times infinity.
    used = source > 0
    lower = float(source[used] @ divergence[used])
    upper = float(np.max(divergence))
    return source, divergence, lower, upper

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
            if PX[i] == 0: continue # zero-probability terms contribute zero
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
            if PY[j] == 0: continue # zero-probability terms contribute zero
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
        """Optimize finite-channel capacity using Blahut-Arimoto iterations.
        Stop when the capacity bounds differ by at most 1e-12 bits.
        Raise RuntimeError if 100,000 iterations do not establish that bound.
        The instance's source probabilities are preserved.
        """
        P = self.__P
        # An input-independent output carries no information, exactly.
        if np.all(P == P[0]):
            capacity = 0.0
            source = np.full(P.shape[0], 1 / P.shape[0])
        else:
            # Work in log probabilities to avoid losing rare inputs to underflow.
            # Omit unreachable outputs, and let zero transition terms contribute zero.
            P = P[:, np.any(P > 0, axis=0)]
            positive = P > 0
            logP = np.full(P.shape, -np.inf)
            np.log(P, out=logP, where=positive)
            log_source = np.full(P.shape[0], -np.log(P.shape[0]))
            log_two = np.log(2.0)
            for iteration in range(_CAPACITY_MAX_ITERATIONS):
                source, divergence, lower, upper = _capacity_bounds(
                    P, logP, positive, log_source)
                if upper - lower <= _CAPACITY_TOLERANCE_BITS * log_two:
                    capacity = max(0.0, lower / log_two)
                    break
                # Boundary optima can converge slowly. Try omitting currently
                # unhelpful inputs, but accept only if the bounds over ALL input
                # rows certify capacity. Otherwise continue the original iteration.
                unhelpful = divergence < lower
                if iteration % 100 == 0 and np.any(unhelpful) and not np.all(unhelpful):
                    candidate = log_source.copy()
                    candidate[unhelpful] = -np.inf
                    candidate -= np.logaddexp.reduce(candidate)
                    candidate_source, _, candidate_lower, candidate_upper = _capacity_bounds(
                        P, logP, positive, candidate)
                    if candidate_upper - candidate_lower <= _CAPACITY_TOLERANCE_BITS * log_two:
                        source = candidate_source
                        capacity = max(0.0, candidate_lower / log_two)
                        break
                log_source += divergence
                log_source -= np.logaddexp.reduce(log_source)
            else:
                raise RuntimeError("Channel capacity optimization did not converge")

        if verbose:
            print('Channel capacity maximizes I(X,Y) over source probabilities.')
            for i, probability in enumerate(source):
                print('P(X=' + str(i) + ') = ' + str(probability))
            print('Cs = Imax(X,Y) = ' + str(capacity) + ' bits')
            print('')
        return capacity

    def ChannelEfficiency(self, verbose = True):
        """Return I(X;Y) / Cs for this source and a finite discrete channel.
        Raises ZeroDivisionError when channel capacity is zero.
        """
        Cs = self.Cs(False)
        if Cs == 0:
            raise ZeroDivisionError("Channel efficiency is undefined at zero capacity")
        IXY = self.HY() - self.HYX()
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
