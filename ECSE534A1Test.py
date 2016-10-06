import unittest
import ECSE534A1 as methods
import numpy as np

class Test_test1(unittest.TestCase):
    def test_createTestMatrix(self):
        matrix_size = 2
        result = methods.createSPDMatrix(matrix_size)
        self.assertEqual(result.shape[0], matrix_size, msg='Row is incorrect size')
        self.assertEqual(result.shape[1], matrix_size, msg='Column is incorrect size')

        matrix_size = 0
        with self.assertRaises(IndexError):
            methods.createSPDMatrix(matrix_size)

    def test_createTestMatrices(self):
        lowerBound = 2
        upperBound = 10
        result = methods.createSPDMatrices(lowerBound, upperBound)
        expectedLength = (upperBound - lowerBound)+1
        self.assertEqual(len(result), expectedLength, msg='Result is incorrect length')

        with self.assertRaises(IndexError):
            methods.createSPDMatrices(1, 10)

        with self.assertRaises(IndexError):
            methods.createSPDMatrices(2, 2)
    
    def test_choleskiSolver(self):
        inputMatrix = np.array(([0,0], [0,0]),dtype=np.float)
        initialVector = np.array([0,0],dtype=np.float).T

        with self.assertRaises(Exception):
            methods.choleskiSolver(inputMatrix, initialVector)

        solutionVector = np.array(([8],[5]), dtype=np.float)
        incidenceMatrix = np.array(([5,4],[4,5]), dtype=np.float)
        testMatrix = np.copy(incidenceMatrix)
        initialVector = incidenceMatrix.dot(solutionVector)
        result = methods.choleskiSolver(incidenceMatrix, initialVector)
        np.testing.assert_allclose(np.tril(incidenceMatrix).dot(np.tril(incidenceMatrix).T), testMatrix)
        np.testing.assert_allclose(result, solutionVector)

        testMatrices = methods.createSPDMatrices(2, 10)
        for testMatrix in testMatrices:
            solutionVector = np.linspace(1,1,testMatrix.shape[0], dtype=np.float)
            initialVector = testMatrix.dot(solutionVector)
            result = methods.choleskiSolver(testMatrix, initialVector)
            np.testing.assert_allclose(result, solutionVector)

    def test_readLinearResistiveNetwork(self):
        A = np.array([[1,-1,0,0,0],[-1,0,1,-1,0],[0,1,-1,0,1]], dtype=np.float)
        J = np.array([0,0,0,0,0],dtype=np.float)
        E = np.array([0,0,0,0,1],dtype=np.float)
        R = np.array([100,100,100,100,10],dtype=np.float)
        fileName = "NodeNetworkOne.csv"
        result = methods.readLinearResistiveNetwork(fileName)

        np.testing.assert_allclose(result[0], A)
        np.testing.assert_allclose(result[1], J)
        np.testing.assert_allclose(result[2], E)
        np.testing.assert_allclose(result[3], R)

    def test_solveLinearResistiveNetwork(self):
        circuits = {'TestCircuit1.csv' : np.array([5]), 'TestCircuit2.csv':np.array([50])}
        for fileName, expected in circuits.items():
            result = methods.solveLinearResistiveNetwork(fileName)
            np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
