import unittest
import ECSE534A1 as methods

class Test_test1(unittest.TestCase):
    def test_createTestMatrix(self):
        matrix_size = 2
        result = methods.createTestMatrix(matrix_size)
        self.assertEqual(result.shape[0], matrix_size, msg='Row is incorrect size')
        self.assertEqual(result.shape[1], matrix_size, msg='Column is incorrect size')

    def test_createTestMatrices(self):
        lowerBound = 2
        upperBound = 10
        result = methods.createTestMatrices(lowerBound, upperBound)
        expectedLength = (upperBound - lowerBound)+1
        self.assertEqual(len(result), expectedLength, msg='Result is incorrect length')

if __name__ == '__main__':
    unittest.main()
