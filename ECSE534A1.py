import numpy as np
import random
import logging

logger = logging.getLogger('numerical-application')

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

def choleskiFacotrization(inputMatrix, initialValueVector):

    #input validation
    if inputMatrix.dtype == np.integer:
        logger.warning('inputMatrix is of type integer, there will be a \
        loss of precision in this algorithm. Please provide float')
    if initialValueVector.dtype == np.integer:
        logger.warning('initialValueVector is of type integer, there will be a \
        loss of precision in this algorithm. Please provide float')

    rowLength = inputMatrix.shape[0]
    columnLength = inputMatrix.shape[1]
    
    for j in range(columnLength):
        if inputMatrix[j, j] <= 0:
            raise Exception("invalid value in matrix, it is not SPD")
        
        sqrtTerm = np.sqrt(inputMatrix[j,j])
        inputMatrix[j,j] = sqrtTerm
        initialValueVector[j] = initialValueVector[j] / inputMatrix[j,j]
        for i in range(j+1, rowLength):
            value = inputMatrix[i,j]
            inputMatrix[i,j] = inputMatrix[i,j] / inputMatrix[j,j]
            initialValueVector[i] = initialValueVector[i] - (inputMatrix[i,j]*initialValueVector[j])
            for k in range(j+1, i+1):
                inputMatrix[i,k] = inputMatrix[i,k] - inputMatrix[i,j]*inputMatrix[k,j]

    return (inputMatrix, initialValueVector)

def backSubstitution(upperTriangularMatrix, intermediateValues):
    resultVector = np.array()

    return resultVector
if __name__ == '__main__':
    
    pass