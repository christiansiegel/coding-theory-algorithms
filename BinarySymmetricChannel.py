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

    __order = float(2)
    __print_output = True
    __ProbabilityMatrix = np.array([[0, 0],
                                    [0, 0]]).astype(float)

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

    @staticmethod
    def Entropy(self):
        """
        Calculates the entropy of the probability matrix
        :return:
        """
        return

    def PostprioriEntropy(self):
        """
        Calculates the post priori entropy of the probability matrix
        :return:
        """

        # Get the backwards probability matrix
        BackwardsProp = self.BackwardsProbability()
        ResultVector = np.empty(np.size(BackwardsProp, 1), float)

        # Calculate the post priory by summing each row of values from the backwards probability
        # Each value gets summed as value + log2(1/value)
        for i in range(0, np.size(BackwardsProp, 1)):
            for j in range(0, np.size(BackwardsProp, 0)):
                ResultVector[j] += BackwardsProp[i, j] * np.log2(1.0 / BackwardsProp[i, j])

        return ResultVector

    def CalcOutputSymbolVector(self):
        """
        Calculates each P(X_i/Y_j) and returns it as a vector
        :return:
        """
        # Get the size of the horizontal axis
        Size = np.size(self.__ProbabilityMatrix, 0)
        # Create an empty array for storage
        Result = np.empty(Size, float)

        # Calculate each output probability
        for i in range(0, Size):
            Result[i] = self.__ProbabilityMatrix[0, i] * 1.0 / float(self.__order) + \
                        self.__ProbabilityMatrix[1, i] * 1.0 / float(self.__order)
        if self.__print_output:
            print(Result)

        return Result

    def BackwardsProbability(self):
        """
        Calculates the backwards probability values of all input/output sets and stores them in a vector where
        0 - Size/2 is x0 and Size/2 to Size is x1 sets.
        :return:
        """
        HorizontalSize = np.size(self.__ProbabilityMatrix, 0)
        VerticalSize = np.size(self.__ProbabilityMatrix, 1)
        ResultMatrix = np.empty([HorizontalSize, VerticalSize])

        OutputSymbols = self.CalcOutputSymbolVector()
        for i in range(0, HorizontalSize):  # Iterate through x
            for j in range(0, VerticalSize):  # Iterate through y
                # (P(y_j/x_i) * P(x_i))/ P(y_j)
                ResultMatrix[i, j] = (self.__ProbabilityMatrix[i, j] * 1.0 / float(self.__order)) / OutputSymbols[j]
        return ResultMatrix