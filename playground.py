from BinarySymmetricChannel import BinarySymmetricChannel
from LinearBlockCode import LinearBlockCode
from CyclicCode import CyclicCode
import numpy as np

if __name__ == '__main__':
    G = np.array([[1,1,0,1,0,0,0],
                  [0,1,1,0,1,0,0],
                  [1,1,1,0,0,1,0],
                  [1,0,1,0,0,0,1]])
    code = LinearBlockCode()
    code.setG(G)
    code.printInfo()
