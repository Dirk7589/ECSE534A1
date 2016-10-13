import numpy as np
import random
import logging
import csv
import time
import matplotlib.pyplot as plt

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

def choleskiSolverSparse(inputMatrix, initialValueVector, halfBandwidth):
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

        bandwidthLimit = j + halfBandwidth + 1
        if bandwidthLimit > columnLength:
            bandwidthLimit = columnLength #Ensure we don't index past the bottom

        for i in np.arange(j+1, bandwidthLimit): #Iterate over those items in the band
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
    '''Generates a n by m mesh
    :param n: the height of the mesh
    :param m: the width of the mesh
    Returns the resulting incidence matrix for the mesh
    The following number scheme for nodes and elements is used:
    Nodes number 0, 1, 2, ... along the horizontal and elements number 
    0, 1, 2, ... where an element to the right of node is numbered first
    and an element beneth the node is numbered second.
    E.g. 2x2 generates, note a, b, c, d represent the elements:
    0, a, 1
    b, , c
    2, d, 3
    Also current is considered to be flowing to the right and down through every element.
    Note, sources must respect this current convention
    '''
    result = []
    numberOfNodes = n * m
    numberofBranches = n*(m-1) + (n-1)*m
    incidenceMatrix = np.zeros((numberOfNodes,numberofBranches))
    E = np.zeros(numberofBranches)
    J = np.zeros(numberofBranches)
    R = np.ones(numberofBranches)
    R = R*1000

    currentBranchElement = 0 #Note the current branch element
    for i in np.arange(n): #Loop through the rows
        for j in np.arange(m): #Loop through the columns
            currentNodeIndex = m*i + j #Compute the current node number given our scheme

            #Compute the current flow for any node not in the last column or row
            if i != (n-1) and j != (m-1):
                incidenceMatrix[currentNodeIndex, currentBranchElement] = 1 
                incidenceMatrix[currentNodeIndex + 1, currentBranchElement] = -1
                incidenceMatrix[currentNodeIndex, currentBranchElement + 1] = 1
                incidenceMatrix[currentNodeIndex + m, currentBranchElement + 1] = -1

                currentBranchElement = currentBranchElement + 2 #Handled two branches

            #Handle the last column
            elif i != (n-1) and j == (m-1):
                if i == 0:
                    incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                    incidenceMatrix[currentNodeIndex + m, currentBranchElement] = -1
                else:
                    incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                    incidenceMatrix[currentNodeIndex + m, currentBranchElement] = -1

                currentBranchElement = currentBranchElement + 1 #Handled one branch
            #Handle the last row
            elif i == (n-1) and j != (m-1):
                incidenceMatrix[currentNodeIndex, currentBranchElement] = 1
                incidenceMatrix[currentNodeIndex + 1, currentBranchElement] =  -1

                currentBranchElement = currentBranchElement + 1 #Handled one branch
            
            else:
                #incidenceMatrix[0, currentBranchElement - 1] = 1 
                #incidenceMatrix[currentNodeIndex, currentBranchElement - 1] = -1
                pass
    lowerLeftNodeIndex = m*(n-1)
    lowerLeftNode = incidenceMatrix[lowerLeftNodeIndex]
    for j in lowerLeftNode:
        if lowerLeftNode[j] != 0:
            incidenceMatrix[m-1,j] = lowerLeftNode[j]
    np.delete(incidenceMatrix, lowerLeftNodeIndex)
    result.append(incidenceMatrix)
    result.append(E)
    result.append(J)
    result.append(R)
    return result

def solveMeshResistances(sparse=False):
    meshSizes = np.linspace(2,15,1)
    results = []
    for meshSize in meshSizes:
        deltaTime = 0
        equivalentResistance = 0
        result = {'size':0, 'time':0, 'req':[]}
        elements = meshWriter(meshSize, meshSize)
        A = elements[0]
        E = elements [1]
        J = elements[2]
        Y = np.diag(1/elements[3])
        
        startTime = time.time() #start timing
        inputMatrix = A.dot(Y).dot(A.T)
        initialVector = A.dot((J-Y.dot(E)))
        if sparse:
            b = A.shape[0]
            solutionVector = choleskiSolverSparse(inputMatrix, initialVector, b)
        else:
            solutionVector = choleskiSolver(inputMatrix, initialVector)
        #Compute Req

        deltaTime = time.time() - startTime #end timing
        #Assign results
        result['size'] = meshSize
        result['time'] = deltaTime
        result['req'] = equivalentResistance
        results.append(result)
    return result

def runLinearResistiveMeshTests():
    nonSparseResults = solveMeshResistances(False)
    sparseResults = solveMeshResistances(True)
    sizeVReq = [[],[]]
    for result in nonSparseResults:
        sizeVReq[0] = result['size']
        sizeVReq[1] = result['req']
    plt.plot(sizeVReq[0], sizeVReq[1])

def sorSolver(inputMatrix, relaxation, tolerance):
    if relaxation > 2:
        raise Exception('You have an illegal relaxation value {},\
        it must be less than 2'.format(relaxation))
    residual = 1
    iterationNumber = 0
    updatedInputMatrix = inputMatrix #Set the initial guess to our current input

    return 
if __name__ == '__main__':
    pass
