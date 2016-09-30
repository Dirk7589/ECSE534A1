import numpy as np
import random
import logging

logger = logging.getLogger('numerical-application')

def createTestMatrices(lowerBound, upperBound):
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
        testMatrices.append(createTestMatrix(i))
    return testMatrices

def createTestMatrix(size):
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

def choleskiFacotrization(A, b):

    #input validation
    if A.dtype == np.integer:
        logger.warning('inputMatrix is of type integer, there will be a \
        loss of precision in this algorithm. Please provide float')
    if b.dtype == np.integer:
        logger.warning('initialValueVector is of type integer, there will be a \
        loss of precision in this algorithm. Please provide float')

    rowLength = A.shape[0]
    columnLength = A.shape[1]
    L = np.zeros((rowLength,columnLength), dtype=np.float)
    for j in range(columnLength):
        if A[j, j] <= 0:
            raise Exception("invalid value in matrix, it is not SPD")
        
        sqrtTerm = np.sqrt(A[j,j])
        L[j,j] = sqrtTerm
        b[j] = b[j] / L[j,j]
        for i in range(j+1, rowLength):
            value = A[i,j]
            L[i,j] = A[i,j] / L[j,j]
            b[i] = b[i] - (L[i,j]*b[j])
            for k in range(j+1, i+1):
                A[i,k] = A[i,k] - L[i,j]*L[k,j]

    return (L, b)
if __name__ == '__main__':
    
    result = createTestMatrices(2, 12)
    nonSPDMatrix = np.array([(-2,1), (1,2)])
    result = choleskiFacotrization(None, None)
    print(result)
    #print(nonSPDMatrix)