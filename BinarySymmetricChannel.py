# -*- coding: UTF-8 -*-

import numpy as np
import math


class BinarySymmetricChannel:
    """Binary Symmetric Channel

    Based on the the Binary Symmetric Channel lectures (date ---)
    slides ---.

    Attributes:
        :param: __order: Defaults to 2, order variable to make code able to handle non-binary inputs in the future
        :param: __ProbabilityMatrix: The probability matrix of the Binary Symmetric Channel
        :param: __print_output: If true, prints verbose output information

    """

    def __init__(self):
        pass

    __order = 2.0
    __print_output = True
    __ProbabilityMatrix = np.zeros([2, 2], float)

    def SetProbabilityMatrix(self, ProbabilityMatrix):
        """
        Replaces the current __ProbabilityMatrix
        :param ProbabilityMatrix: New Probability matrix
        :return:
        """
        self.__ProbabilityMatrix = np.array(ProbabilityMatrix).astype(float)

    def GetProbabilityMatrix(self):
        """
        Gets the ___ProbabilityMatrix
        :return: ___ProbabilityMatrix
        """
        return self.__ProbabilityMatrix

    def SetPrintOutput(self, flag):
        """
        Sets if the code should print verbose output
        :param flag: True prints verbose output
        :return:
        """
        self.__print_output = flag


    def Entropy(self):
        """
        Calculates the entropy of the probability matrix
        :return:
        """
        return 0.0

    def PostprioriEntropy(self):
        """
        Calculates the post priori entropy of the probability matrix
        :return:
        """

        # Get the backwards probability matrix
        BackwardsProp = self.BackwardsProbability()
        HorizontalSize = np.size(BackwardsProp, 0)
        VerticalSize = np.size(BackwardsProp, 1)
        ResultVector = np.zeros(VerticalSize)
        # print(ResultVector)
        # print(VerticalSize)
        # print(HorizontalSize)

        # Calculate the post priory by summing each row of values from the backwards probability
        # Each value gets summed as value + log2(1/value)
        for i in range(0, VerticalSize):
            for j in range(0, HorizontalSize):
                ResultVector[j] += BackwardsProp[i, j] * np.log2(1.0 / BackwardsProp[i, j])

        return ResultVector

    @property
    def CalcOutputSymbolVector(self):
        """
        Calculates each P(X_i/Y_j) and returns it as a vector
        :return:
        """
        # Get the size of the horizontal axis
        Size = np.size(self.__ProbabilityMatrix, 0)
        # Create an empty array for storage
        Result = np.zeros(Size, float)

        # Calculate each output probability
        for i in range(0, Size):
            Result[i] = self.__ProbabilityMatrix[0, i] * (1.0 / float(self.__order)) + \
                        self.__ProbabilityMatrix[1, i] * (1.0 / float(self.__order))

        # if self.__print_output:
            # print(Result)

        return Result

    def BackwardsProbability(self):
        """
        Calculates the backwards probability values of all input/output sets and stores them in a vector where
        0 - Size/2 is x0 and Size/2 to Size is x1 sets.
        :return:
        """
        HorizontalSize = np.size(self.__ProbabilityMatrix, 0)
        VerticalSize = np.size(self.__ProbabilityMatrix, 1)
        ResultMatrix = np.zeros([HorizontalSize, VerticalSize],float)

        OutputSymbols = self.CalcOutputSymbolVector
        for i in range(0, VerticalSize):  # Iterate through x
            for j in range(0, HorizontalSize):  # Iterate through y
                # (P(y_j/x_i) * P(x_i))/ P(y_j)
                ResultMatrix[i, j] = (self.__ProbabilityMatrix[i, j] * (1.0 / self.__order)) / OutputSymbols[j]

        return ResultMatrix

if __name__ == '__main__':

    np.set_printoptions(precision=4)
    Code = BinarySymmetricChannel()
    Code.SetPrintOutput(True)
    prob = np.array([[3.0/5.0, 2.0/5.0],
                     [1.0/5.0, 4.0/5.0]])

    # Test probability array
    Code.SetProbabilityMatrix(prob)
    print(Code.PostprioriEntropy())
