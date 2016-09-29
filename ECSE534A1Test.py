import unittest
import ECSE534A1 as methods
import numpy as np

class Test_test1(unittest.TestCase):
    def test_createTestMatrix(self):
        matrix_size = 2
        result = methods.createTestMatrix(matrix_size)
        self.assertEqual(result.shape[0], matrix_size, msg='Row is incorrect size')
        self.assertEqual(result.shape[1], matrix_size, msg='Column is incorrect size')

        matrix_size = 0
        with self.assertRaises(IndexError):
            methods.createTestMatrix(matrix_size)

    def test_createTestMatrices(self):
        lowerBound = 2
        upperBound = 10
        result = methods.createTestMatrices(lowerBound, upperBound)
        expectedLength = (upperBound - lowerBound)+1
        self.assertEqual(len(result), expectedLength, msg='Result is incorrect length')

        with self.assertRaises(IndexError):
            methods.createTestMatrices(1, 10)

        with self.assertRaises(IndexError):
            methods.createTestMatrices(2, 2)

    def test_choleskiFacotrization(self):
        inputMatrix = np.array(([0,0], [0,0]))
        initialVector = np.array([0,0]).T

        with self.assertRaises(Exception):
            methods.choleskiFacotrization(inputMatrix, initialValueVector)

if __name__ == '__main__':
    unittest.main()
