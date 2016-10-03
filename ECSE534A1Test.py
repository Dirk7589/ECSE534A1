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
        testMatrices = methods.createSPDMatrices(2,10)
        for testMatrix in testMatrices:
            size = testMatrix.shape[0]
            solutionVector = np.linspace(1, 1, num=size, dtype=np.float)
            result = methods.choleskiSolver(testMatrix, solutionVector)
            np.testing.assert_allclose(result, solutionVector)

    def test_backSubstitution(self):
        #setup
        size = 2
        solutionVector = np.array(([2],[1]), dtype=np.float)
        incidenceMatrix = np.array(([5,4],[4,5]), dtype=np.float)
        initialVector = incidenceMatrix.dot(solutionVector)
        choleskiResult = methods.choleskiFacotrization(incidenceMatrix, initialVector)
        #run
        result = methods.backSubstitution(choleskiResult[0].T, choleskiResult[1])
        #assert
        np.testing.assert_allclose(result, solutionVector)
    
    def test_choleskiFacotrization(self):
        inputMatrix = np.array(([0,0], [0,0]),dtype=np.float)
        initialVector = np.array([0,0],dtype=np.float).T

        with self.assertRaises(Exception):
            methods.choleskiFacotrization(inputMatrix, initialVector)

        testMatrix = np.array(([25,15,-5],[15,18,0],[-5,0,11]),dtype=np.float)
        inputMatrix = np.copy(testMatrix)
        initialVector = np.array([1,1,1], dtype=np.float).T
        result = methods.choleskiFacotrization(inputMatrix, initialVector)
        expected = result[0].dot(result[0].T)
        np.testing.assert_equal(testMatrix, expected)


if __name__ == '__main__':
    unittest.main()
