import numpy as np
import random
import logging
import csv

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
    return testMatrix.astype(np.float)

def choleskiSolver(inputMatrix, initialValueVector):
    """"Solves a matrix problem inputMatrix*x=initialValueVector using choleski
    decomposition.
    :param inputMatrx: The square SPD input matrix
    :param initialValueVector: The vector containing the initial conditions
    Returns the resulting x vector
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
            initialValueVector[i] = initialValueVector[i] - \
                (inputMatrix[i,j]*initialValueVector[j])
            for k in np.arange(j+1, i+1):
                inputMatrix[i,k] = inputMatrix[i,k] - inputMatrix[i,j]*inputMatrix[k,j]
    
    resultVector = np.zeros_like(initialValueVector)
    n = resultVector.shape[0]
    i = n - 1
    while(i >= 0):
        sum = 0
        for j in np.arange(i+1, n):
            sum += (inputMatrix[j,i]*resultVector[j])
        resultVector[i] = (initialValueVector[i] - sum) / inputMatrix[i,i]
        i -= 1
    return resultVector

def readLinearResistiveNetwork(fileName):
    
    A = []
    J = []
    E = []
    R = []
    aMode = False
    jMode = False
    eMode = False
    rMode = False
    with open(fileName) as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        for row in reader:
            if "A" in row:
                aMode = True
                jMode = False
                eMode = False
                rMode = False
                continue
            elif "J" in row:
                aMode = False
                jMode = True
                eMode = False
                rMode = False
                continue
            elif "E" in row:
                aMode = False
                jMode = False
                eMode = True
                rMode = False
                continue
            elif "R" in row:
                aMode = False
                jMode = False
                eMode = False
                rMode = True
                continue
            if aMode:
                A.append(row)
            elif jMode:
                J = row
            elif eMode:
                E = row
            elif rMode:
                R = row
    
    return [np.array(A,dtype=np.float),
            np.array(J,dtype=np.float).T,
            np.array(E,dtype=np.float).T,
            np.array(R,dtype=np.float).T]

def solveLinearResistiveNetwork(fileName):
    elements = readLinearResistiveNetwork(fileName)
    A = elements[0]
    J = elements [1]
    E = elements[2]
    Y = np.diag(1/elements[3])

    inputMatrix = A.dot(Y).dot(A.T)
    initialVector = A.dot((J-Y.dot(E)))
    result = choleskiSolver(inputMatrix, initialVector)
    return result

def meshWriter(n, m):
        numberOfNodes = n * m
        numberofBranches = n*(m-1) + (n-1)*m
        incidenceMatrix = np.zeros((numberOfNodes,numberofBranches))

        currentBranchElement = 0
        for i in np.arange(n):
            for j in np.arange(m):
                currentNodeIndex = m*i + j

                if i != (n-1) and j != (m-1):
                    incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                    incidenceMatrix[currentNodeIndex + 1, currentBranchElement] = -1
                    incidenceMatrix[currentNodeIndex, currentBranchElement + 1] = 1
                    incidenceMatrix[currentNodeIndex + m, currentBranchElement + 1] = -1

                    currentBranchElement = currentBranchElement + 2
                elif i != (n-1) and j == (m-1):
                    incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                    incidenceMatrix[currentNodeIndex + m, currentBranchElement] = -1

                    currentBranchElement = currentBranchElement + 1
                elif i == (n-1) and j != (m-1):
                    incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                    incidenceMatrix[currentNodeIndex + 1, currentBranchElement] =  -1

                    currentBranchElement = currentBranchElement + 1
                else:
                    incidenceMatrix[0, currentBranchElement - 1] = 1 
                    incidenceMatrix[currentNodeIndex, currentBranchElement - 1] = -1

        return incidenceMatrix
if __name__ == '__main__':
    result = solveLinearResistiveNetwork("NodeNetworkOne.csv")
    pass
