﻿import numpy as np
import random
import logging

def initLogger():
    logger = logging.getLogger('numerical-application')
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    return logger
logger =  initLogger()

def createSPDMatrices(lowerBound, upperBound):
    """Computes a series of test matrices for Choleski algorithm
    :param lowerBound: dimension of smallest test matrix
    :param upperBound: dimension of largest test matrix
    Return the test matrices
    """
    if lowerBound <= 1:
        raise IndexError('lowerBound must be larger than 1')
    elif upperBound <= lowerBound:
        raise IndexError('upperBound must be larger than lowerBound')
    testMatrices = []
    for i in range(lowerBound, upperBound + 1):
        testMatrices.append(createSPDMatrix(i))
    return testMatrices

def createSPDMatrix(size):
    """Creates an nxn matrix that is SPD with no zeros
    Return the test matrix
    """
    if size <= 1:
        raise IndexError('unsupported size {} must be larger than 1'.format(size))
    
    testMatrix = None
    aMatrix = np.random.randint(1,9,size=(size, size)) #generate random matrix
    bMatrix = np.transpose(aMatrix) #compute the transpose
    testMatrix = aMatrix.dot(bMatrix) #generate an SPD matrix
    return testMatrix

def choleskiSolver(inputMatrix, initialValueVector):
    """"Solves a matrix problem inputMatrix*x=initialValueVector using choleski
    decomposition.
    :param inputMatrx: The square SPD input matrix
    :param initialValueVector: The vector containing the initial conditions
    Returns the resulting x vector
    """
    choleskiResult = choleskiFacotrization(inputMatrix, initialValueVector)
    result = backSubstitution(choleskiResult[0], choleskiResult[1])
    return result


def choleskiFacotrization(inputMatrix, initialValueVector):
    """Computes the choelski factorization along with the intermediate values
    :param inputMatrix: matrix to be converted to lower triangular
    :param initialValueVector: initital value vector
    Returns a tuple, (lower triangular result, intermediate values)
    """
    #input validation
    if inputMatrix.dtype == np.integer:
        logger.warning('inputMatrix is of type integer.\
        there will be a loss of precision. Please provide float')
    if initialValueVector.dtype == np.integer:
        logger.warning('initialValueVector is of type integer.\
        there will be a loss of precision. Please provide float')

    rowLength = inputMatrix.shape[0]
    columnLength = inputMatrix.shape[1]
    
    for j in np.arange(columnLength):
        if inputMatrix[j, j] <= 0:
            raise Exception("invalid value in matrix, it is not SPD")
        
        sqrtTerm = np.sqrt(inputMatrix[j,j])
        inputMatrix[j,j] = sqrtTerm
        initialValueVector[j] = initialValueVector[j] / inputMatrix[j,j]
        for i in np.arange(j+1, rowLength):
            value = inputMatrix[i,j]
            inputMatrix[i,j] = inputMatrix[i,j] / inputMatrix[j,j]
            initialValueVector[i] = initialValueVector[i] - (inputMatrix[i,j]*initialValueVector[j])
            for k in np.arange(j+1, i+1):
                inputMatrix[i,k] = inputMatrix[i,k] - inputMatrix[i,j]*inputMatrix[k,j]
    
    inputMatrix = np.tril(inputMatrix) #zeroes items above the diagonal
    return (inputMatrix, initialValueVector)

def backSubstitution(upperTriangularMatrix, inputVector):
    """Performs the back substitution on an upper triangular matrix,
    for a given inputVector
    :param upperTriangularMatrix: the upper triangular matrix to solve
    :param inputVector: the associated input vector to substitute
    Returns the resulting vector
    """
    resultVector = np.zeros_like(inputVector)
    n = resultVector.shape[0]
    i = n
    while(i > 0):
        i -= 1
        sum = 0
        for j in np.arange(i+1, n):
            newValue = upperTriangularMatrix[i,j]*resultVector[j]
            sum = sum + newValue
        resultVector[i] = (inputVector[i] - sum) / upperTriangularMatrix[i,i]

    return resultVector 

if __name__ == '__main__':
    
    pass