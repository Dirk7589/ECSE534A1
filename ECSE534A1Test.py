﻿import unittest
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
        inputMatrix = np.array(([0,0], [0,0]),dtype=np.float)
        initialVector = np.array([0,0],dtype=np.float).T

        with self.assertRaises(Exception):
            methods.choleskiFacotrization(inputMatrix, initialVector)
        testMatrices = methods.createTestMatrices(2,10)
        for testMatrix in testMatrices:
            testMatrix = np.array(([5,4],[4,5]),dtype=np.float)
            inputMatrix = np.copy(testMatrix)
            initialVector = np.array([1,1], dtype=np.float).T
            result = methods.choleskiFacotrization(inputMatrix, initialVector)
            expected = result[0].dot(result[0].T)
            self.assertEquals(testMatrix.all(), expected.all())

if __name__ == '__main__':
    unittest.main()
