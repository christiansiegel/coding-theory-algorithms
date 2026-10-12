import numpy as np
import math
from itertools import combinations
from .validation import binary_matrix, coefficients, probability


def HtoG(H):
    """Convert a Parity Check Matrix in systematic form to a Generator Matrix.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Generator Matrix G
    """
    n = np.shape(H)[1]
    k = n - np.shape(H)[0]
    P = HtoP(H)
    Ik = np.eye(k)
    G = np.concatenate((P, Ik), axis=1)
    return G.astype(int)


def GtoH(G):
    """Convert a Generator Matrix in systematic form to a Parity Check Matrix.
    Args:
        G: Generator Matrix in systematic form
    Returns:
        Parity Check Matrix H
    """
    k = np.shape(G)[0]
    n = np.shape(G)[1]
    P = GtoP(G)
    PT = np.transpose(P)
    Ik = np.eye(n - k)
    H = np.concatenate((Ik, PT), axis=1)
    return H.astype(int)


def GtoP(G):
    """Extract the submatrix P from a Generator Matrix in systematic form.
    Args:
        G: Generator Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    k = np.shape(G)[0]
    n = np.shape(G)[1]
    P = G[:k, : n - k]
    return P.astype(int)


def HtoP(H):
    """Extract the submatrix P from a Parity Check Matrix in systematic form.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    n = np.shape(H)[1]
    k = n - np.shape(H)[0]
    PK = H[:, n - k : n]
    P = np.transpose(PK)
    return P.astype(int)


def matrixMultiplicationEquations(M, aSymbol, bSymbol):
    """Symbolic matrix multiplication.

        a = b * M

    Where   M: binary matrix
            b: symbolic vector with symbol 'bSymbol'
            a: symbolic vector with symbol 'aSymbol'

    Example:
        M = [[0,1,1],
             [1,1,0],
             [1,1,1]]
        aSymbol = 'a'
        bSymbol = 'b'

        Resulting String:
            a0 = b1 + b2
            a1 = b0 + b1 + b2
            a2 = b0 + b2
    """
    k = np.shape(M)[0]
    n = np.shape(M)[1]
    equations = ""
    for i in range(0, n):
        s = aSymbol + str(i) + " = "
        m = []
        for j in range(0, k):
            if M[j, i] == 1:
                m.append(bSymbol + str(j))
        s += " \u2295 ".join(m)
        equations += "\n" + s
    return equations


def w(v):
    """Hamming weight of a vector (slide 52)
    Args:
        v: vector
    Returns:
        Hamming weight of the vector
    """
    return np.count_nonzero(v)


def d(v1, v2):
    """Hamming distance of two vectors (slide 53)
    Args:
        v1: vector 1
        v2: vector 2
    Returns:
        Hamming distance of the vectors
    """
    return w((v1 + v2) % 2)


def intToArray(i, length=0):
    """Convert an unsigned integer to a least-significant-bit-first array.
    Args:
        i: unsigned integer
        length: padding to length (default: 0)
    Returns:
        uint8 binary array, with zero-padding at the end if length is given.
    """
    if length > 0:
        s = np.binary_repr(i, width=length)
    else:
        s = np.binary_repr(i)
    m = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    m = np.flipud(m)
    return m


def arrayToString(a):
    """Convert an array of integer numbers to a string.
    Args:
        a: array of integer numbers
    Returns:
        string representation of array
    """
    s = ""
    for i in range(len(a)):
        s += str(int(a[i]))
    return s


def nCr(n, k):
    """Return the exact integer binomial coefficient for nonnegative n and k."""
    return math.comb(n, k)


class LinearBlockCode:
    """Linear Block Code

    Based on the the Linear Block Codes lecture (2016-11-02)
    slides 30-76.

    Attributes:
        __G: The Generator Matrix of the Linear Block Code
    """

    __G = np.empty([0, 0])

    def k(self):
        """Message length in bits."""
        return np.shape(self.G())[0]

    def n(self):
        """Codeword length in bits."""
        return np.shape(self.G())[1]

    def R(self):
        """Coding rate (R = k/n)."""
        return self.k() / self.n()

    def G(self):
        """Generator Matrix of the Linear Block Code."""
        return self.__G

    def setG(self, G):
        """Set Generator Matrix of the Linear Block Code.
        Args:
            G: Generator Matrix
        """
        self.__G = binary_matrix(G, "G", generator=True)

    def P(self):
        """Submatrix P of the generator matrix in systematic form."""
        P = GtoP(self.G())
        return P.astype(int)

    def H(self):
        """Parity Check Matrix of the Linear Block Code."""
        H = GtoH(self.G())
        return H.astype(int)

    def setH(self, H):
        """Set Parity Check Matrix of the Linear Block Code.
        Args:
            H: Parity Check Matrix
        """
        G = HtoG(binary_matrix(H, "H", generator=False))
        self.__G = G.astype(int)

    def c(self, m):
        """Generate codeword of a message.
        Args:
            m: message
        Returns:
            codeword
        """
        m = coefficients(m, "m", length=self.k())
        c = m.dot(self.G()) % 2
        return c.astype(int)

    def s(self, r):
        """Generate the syndrome vector (slide 44)
        Args:
            r: Either a received message vector r or an error vector e.
        Returns:
            Syndrome vector
        """
        r = coefficients(r, "r", length=self.n())
        HT = np.transpose(self.H())
        s = r.dot(HT) % 2
        return s.astype(int)

    def M(self):
        """Matrix of all messages."""
        k = self.k()
        M = np.empty([2**k, k])
        for i in range(0, 2**k):
            M[i] = intToArray(i, k)
        return M.astype(int)

    def C(self):
        """Matrix of all codewords."""
        n = self.n()
        k = self.k()
        C = np.empty([2**k, n])
        for i in range(0, 2**k):
            m = intToArray(i, k)
            c = self.c(m)
            C[i] = c
        return C.astype(int)

    def dmin(self, Verbose=False):
        """
        Minimum distance of a linear block code (slide 55)
        """
        dmin = self.n()
        M = self.M()
        if Verbose:
            print("We start by selecting dmin = n")
            print("dmin = ", dmin)
            print("Iterate through the code table and compare the weight of each code vectors")
        for m in M:
            c = self.c(m)
            if w(c) != 0 and w(c) < dmin:
                if Verbose:
                    print(
                        "the weight of ",
                        c,
                        " is ",
                        w(c),
                        " < ",
                        dmin,
                        " we update dmin, dmin = ",
                        w(c),
                    )
                dmin = w(c)
            else:
                if Verbose:
                    print(
                        "Vector ",
                        c,
                        " has a weight of ",
                        w(c),
                        " and isn't a better choice",
                    )

        return dmin

    def dminVerbose(self):
        self.dmin(True)

    def errorDetectionCapability(self):
        """Error Detection Capability of the Block Code (slide 60)."""
        return self.dmin() - 1

    def t(self):
        """Error Correction Capability of a Block Code (slide 64)."""
        return math.floor((self.dmin() - 1) / 2)

    def PU(self, p):
        """Probability of a nonzero undetected error under independent bit flips.
        Args:
            p: bit-flip probability in [0, 1].
        """
        p = probability(p)
        PU = 0
        n = self.n()
        for i in range(1, n + 1):
            PU += self.Ai(i) * p**i * (1 - p) ** (n - i)
        return PU

    def Pe(self, p):
        """Probability of more than t independent bit errors.

        This is the probability of exceeding the guaranteed correction radius.
        Decoder behavior beyond that radius is not specified by this method.
        Args:
            p: bit-flip probability in [0, 1].
        """
        p = probability(p)
        Pe = 0
        n = self.n()
        t = self.t()
        for i in range(t + 1, n + 1):
            Pe += nCr(n, i) * p**i * (1 - p) ** (n - i)
        return Pe

    def Ai(self, i):
        """Weight distribution = number of codewords of weight i (slide 61)
        Args:
            i: weight
        """
        C = self.C()
        A = 0
        for c in C:
            if w(c) == i:
                A += 1
        return A

    def A(self):
        """Weight distribution(slide 61)
        Returns:
            n-entry array: entry j counts codewords of weight j + 1.
            Weight zero is omitted; use Ai(0) for its count.
        """
        n = self.n()
        A = np.empty([n])
        for i in range(0, n):
            A[i] = self.Ai(i + 1)
        return A.astype(int)

    def printMessageCodewordTable(self):
        """Print all messages and their corresponding codewords."""
        M = self.M()
        print("Messages -> Codewords (c = m \u25e6 G)")
        for m in M:
            c = self.c(m)
            print(m, c)

    def printParityCheckEquations(self):
        """Print the parity check equations for the linear block (slide 39)"""
        G = self.G()
        equations = matrixMultiplicationEquations(G, "c", "m")
        print(equations)

    def printSyndromeVectorEquations(self):
        """Print the syndrome vector equations."""
        HT = np.transpose(self.H())
        equations = matrixMultiplicationEquations(HT, "s", "r")
        print(equations)

    def printErrorsThatHaveSyndrome(self, s):
        """Print all error vectors that have the syndrome s.
        Args:
            s: Syndrome vector
        """
        requested_syndrome = np.asarray(s)
        n = self.n()
        print("e0 e1 e2 ... -> weight")
        for i in range(0, 2**n):
            e = intToArray(i, n)
            actual_syndrome = self.s(e)
            if np.array_equal(actual_syndrome, requested_syndrome):
                print(e, "->", w(e))

    def printStandardArray(self):
        """Print every coset once, with a minimum-weight leader in its first column."""
        k = self.k()
        n = self.n()
        codewords = self.C()
        seen_syndromes = set()
        for weight in range(n + 1):
            for positions in combinations(range(n), weight):
                leader = np.zeros(n, dtype=int)
                leader[list(positions)] = 1
                syndrome = tuple(self.s(leader))
                if syndrome in seen_syndromes:
                    continue
                seen_syndromes.add(syndrome)
                row = [arrayToString((codeword + leader) % 2) for codeword in codewords]
                print(row[0] + " | " + " ".join(row[1:]))
                if len(seen_syndromes) == 1:
                    print("-" * ((2**k) * (n + 1) + 1))
                if len(seen_syndromes) == 2 ** (n - k):
                    return

    def correctableErrorPatterns(self):
        """Array of all correctable error patterns."""
        E = np.empty([2 ** np.shape(self.H())[0], self.n()])
        n = self.n()
        t = self.t()
        count = 0
        for i in range(0, 2**n):
            e = intToArray(i, n)
            if w(e) <= t and count < E.shape[0]:
                self.s(e)
                E[count] = e
                count += 1
        E = E[:count]
        return E.astype(int)

    def printDecodingTable(self):
        """Print Decoding Table, which is all correctable error
        patterns and their corresponding unique syndrome vectors.
        (slide 74)
        """
        errors = self.correctableErrorPatterns()
        print("Correctable Error Patterns -> Syndromes")
        for e in errors:
            s = self.s(e)
            print(e, s)

    def decodingTable(self):
        """Decoding table, consisting of a dictionary with all
        correctable error patterns indexed with their unique syndrome
        vector formatted as string.
        e.g.
        "001" -> [0,1,1,0,0,0]
        "011" -> [0,0,1,0,1,0]
        """
        table = {}
        errors = self.correctableErrorPatterns()
        for e in errors:
            s = self.s(e)
            table[arrayToString(s)] = e
        return table

    def syndromeDecode(self, r):
        """Decodes received vector r using syndrome decoding.
        Args:
            r: received vector
        Returns:
            codeword c
        """
        table = self.decodingTable()
        s = self.s(r)
        e = table[arrayToString(s)]
        c = (r + e) % 2
        return c

    def verboseSyndromeDecode(self, r):
        """Decodes received vector r using syndrome decoding.
        Prints all steps of the decoding incl. the whole decoding table.
        Args:
            r: received vector
        Returns:
            codeword c
        """
        print("Decoding received vector r =", r)
        s = self.s(r)
        print("s = r * H' =", s)
        print("Look up the decoding table:")
        print()
        self.printDecodingTable()
        print()
        table = self.decodingTable()
        e = table[arrayToString(s)]
        print("-> find error pattern e =", e)
        c = (r + e) % 2
        print("c = r + e =", c)
        return c

    def printInfo(self):
        """Prints complete Block Code Info"""
        print("-> Linear Block Code Cb(", self.n(), ",", self.k(), ")")
        print("-> Message length (k):             ", self.k())
        print("-> Codeword length (n):            ", self.n())
        print("-> Coding rate (R = k/n):          ", self.R())
        print("-> Minimum Distance (dmin):        ", self.dmin())
        print("-> Error Detection Capability:     ", self.errorDetectionCapability())
        print("-> Error Correction Capability (t):", self.t())
        print("-> Weight Distribution (A):        ", self.A())
        print("-> Generator Matrix (G):")
        print("")
        print(self.G())
        print("")
        print("-> Parity Check Matrix (H):")
        print("")
        print(self.H())
        print("")
        print("-> Message Codeword Table:")
        print("")
        self.printMessageCodewordTable()
        print("")
        print("-> Parity Check Equations:")
        self.printParityCheckEquations()
        print("")
        print("-> Syndrome Vector Equations:")
        self.printSyndromeVectorEquations()
        print("")
        print("-> Standard Array:")
        print("")
        self.printStandardArray()
        print("")
        print("-> Decoding Table:")
        print("")
        self.printDecodingTable()
