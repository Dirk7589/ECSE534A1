﻿import unittest
import ECSE534A1 as methods
import numpy as np
import os
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
        #Setup: Check for positive definite error check
        inputMatrix = np.array(([0,0], [0,0]),dtype=np.float)
        initialVector = np.array([0,0],dtype=np.float).T

        with self.assertRaises(Exception):
            #Run and check an exception was raised
            methods.choleskiSolver(inputMatrix, initialVector) 

        #Setup: Check for decompositiong an solution using known result
        solutionVector = np.array(([8],[5]), dtype=np.float) #Arbitrary solution vector
        incidenceMatrix = np.array(([5,4],[4,5]), dtype=np.float) #Known input matrix
        testMatrix = np.copy(incidenceMatrix) #Store original matrix
        initialVector = incidenceMatrix.dot(solutionVector) #Generate initial vector

        #Run
        result = methods.choleskiSolver(incidenceMatrix, initialVector)
        #Assert
        lowerTriangularMatrix = np.tril(incidenceMatrix)
        #Reconstruct the original matrix
        resultingMatrix = lowerTriangularMatrix.dot(lowerTriangularMatrix.T)
        #Assert the decomposition and solution are correct
        np.testing.assert_allclose(resultingMatrix, testMatrix, 
                                   err_msg='Incorrect decomposition matrix') 
        np.testing.assert_allclose(result, solutionVector, 
                                   err_msg='Incorrect solution vector') 

        #Setup: Check with a series of random matrices
        testMatrices = methods.createSPDMatrices(2, 10)
        for testMatrix in testMatrices:
            solutionVector = np.linspace(1,1,testMatrix.shape[0], dtype=np.float)
            initialVector = testMatrix.dot(solutionVector) #Generate initial vector
            #Run
            result = methods.choleskiSolver(testMatrix, initialVector)
            #Assert
            np.testing.assert_allclose(result, solutionVector)

    def test_CholeskiSolverSparse(self):
        #Setup: Check for positive definite error check
        inputMatrix = np.array(([0,0], [0,0]),dtype=np.float)
        initialVector = np.array([0,0],dtype=np.float).T
        bandwidth = 0

        with self.assertRaises(Exception):
            #Run and check an exception was raised
            methods.choleskiSolverSparse(inputMatrix, initialVector, bandwidth) 

        #Setup: Check for decompositiong an solution using known result
        solutionVector = np.array(([5,3.75,3.75]), dtype=np.float)
        elements = methods.readLinearResistiveNetwork('TestCircuit5.csv')
        A = elements[0]
        J = elements [1]
        E = elements[2]
        Y = np.diag(1/elements[3])

        inputMatrix = A.dot(Y).dot(A.T)
        initialVector = A.dot((J-Y.dot(E)))
        testMatrix = np.copy(inputMatrix) #Store original matrix

        bandwidth = A.shape[0]

        #Run
        result = methods.choleskiSolverSparse(inputMatrix, initialVector, bandwidth)
        #Assert
        np.testing.assert_allclose(result, solutionVector, 
                                   err_msg='Incorrect solution vector') 

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
        #setup
        circuits = {'TestCircuit1.csv' : np.array([5]), 
                    'TestCircuit2.csv':np.array([50]),
                    'TestCircuit3.csv':np.array([55]),
                    'TestCircuit4.csv':np.array([20,35]),
                    'TestCircuit5.csv':np.array([5,3.75,3.75])}
        for fileName, expected in circuits.items():
            error_message = 'Incorrect result for circuit {}'.format(fileName)
            #run
            result = methods.solveLinearResistiveNetwork(fileName)
            #assert
            np.testing.assert_allclose(result, expected, err_msg=error_message)

    def test_runLinearResistiveMeshTests(self):
        #setup
        nonSparseResults = []
        sparseResults = []

        #run
        results = methods.runLinearResistiveMeshTests()

        nonSparseResults = results[0]
        sparseResults = results[1]
        #assert
        for i in np.arange(len(nonSparseResults)):
            np.testing.assert_allclose(nonSparseResults[i]['req'], 
                                       sparseResults[i]['req'])



    def test_meshWriter(self):
        #setup
        expectedIncidence = np.array([[1,1,0,0,1],[-1,0,1,0,0],[0,-1,0,1,0],[0,0,-1,-1,-1]], 
                                     dtype=np.float)
        #run
        result = methods.meshWriter(2, 2)
        np.testing.assert_allclose(result[0], expectedIncidence)
    def test_MeshResistanceSolver(self):
        sparseResult = methods.solveMeshResistances(True)
        nonSparseResult = methods.solveMeshResistances(False)
        for i in range(len(sparseResult)):
            a = sparseResult[i]['req']
            b = nonSparseResult[i]['req']
            np.testing.assert_allclose(a,b)

    def test_relaxationTest(self):
        methods.relaxationTesting()
    def test_meshSizeTesting(self):
        methods.meshSizeTesting()
        
if __name__ == '__main__':
    unittest.main()
