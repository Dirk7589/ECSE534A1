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

def truncateFloat(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

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
    """"Solves a matrix problem inputMatrix*x=initialValueVector using choleski
    decomposition. A sparse varient using bandwidth is used
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
    '''Reads in a linear resistive network
    :param fileName the csv file containing the network to solve
    Returns the A, J, E, and R from the file
    '''
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
    '''Solves a linear resistive network from a file
    :param fileName: The file containing the resistive network
    Returns the solution vector of nodal voltages for the network
    '''
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
    incidenceMatrix = np.zeros((numberOfNodes,numberofBranches+1))
    E = np.zeros(numberofBranches + 1)
    J = np.zeros(numberofBranches + 1)
    R = np.ones(numberofBranches + 1)
    R = R*1000 #Set the resistance to 1000 ohms

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
                #Add a current source to the system
                incidenceMatrix[0, currentBranchElement] = 1 
                incidenceMatrix[currentNodeIndex, currentBranchElement] = -1
                
    J[currentBranchElement] = 1 #Add a current source of 1 Amp
    R[currentBranchElement] = 10**6 #Give the current source a large resistance
    result.append(incidenceMatrix)
    result.append(E)
    result.append(J)
    result.append(R)
    return result

def solveMeshResistances(sparse=False):
    '''Solves a series of linear resistive mesh problems of various size
    :param sparse=False When True, the sparse version of Cholesky is used
    Returns a list of dictionaries that contain the size, time and req'''
    meshSizes = np.arange(2,15)
    results = []
    for meshSize in meshSizes:
        deltaTime = 0
        equivalentResistance = 0
        result = {'size':0, 'time':0, 'req':[]}
        elements = meshWriter(meshSize, meshSize)
        A = elements[0]
        
        A = np.delete(A,len(A)-1, 0)
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
        equivalentResistance = solutionVector[0]
        deltaTime = time.time() - startTime #end timing
        #Assign results
        result['size'] = meshSize
        result['time'] = deltaTime
        result['req'] = equivalentResistance
        results.append(result)
    
    return results #Return all the results

def runLinearResistiveMeshTests():

    nonSparseResults = solveMeshResistances(False)
    #sparseResults = solveMeshResistances(True)
    sizeVReq = [[],[]]
    for result in nonSparseResults:
        print("Mesh size: {}x{} | Computation time: {} | Equivalent R (ohms): {}".format(
            result['size'],result['size'],
            truncateFloat(result['time'], 6),
            truncateFloat(result['req'], 4)))
        sizeVReq[0].append(result['size'])
        sizeVReq[1].append(result['req'])

    
    plt.title('Equivalent resistance (Req) vs mesh size (N)')
    plt.xlabel('N')
    plt.ylabel('Req (ohms)')
    plt.grid(True)
    plt.plot(sizeVReq[0], sizeVReq[1],'-o')
    plt.savefig('q2d.pdf',format='pdf')

def boundaryCheck(corner, innerConductorVoltage, m, n, outerConductorVoltage, potentialMatrix):
    for i in np.arange(n):
        for j in np.arange(m):
            if i == corner[0] and j <= corner[1]:
                if potentialMatrix[i,j] != innerConductorVoltage: #Side f includes corner
                    raise Exception('modified boundary')
            if i == n-1:
                if potentialMatrix[i,j] != outerConductorVoltage: #Side f includes corner
                    raise Exception('modified boundary')
            if j == m-1:
                if potentialMatrix[i,j] != outerConductorVoltage: #Side f includes corner
                    raise Exception('modified boundary')
            if j == corner[1] and i < corner[0]:
                if potentialMatrix[i,j] != innerConductorVoltage: #Side f includes corner
                    raise Exception('modified boundary')

def finiteDifferencePotentialSolver(h, relaxation):
    '''Solves for a set of potentials using finite difference approach.
    By exploiting symmetry, the following region is considered
    ----b--------
    |           |
    c           d
    |---a---|   |
            f   |
            |   |
            --e--
    where c and e are Neuman boundaries and the rest are dirchlet. Note, i, j = (0,0)
    starts in the lower left corner. Note, this formulation uses the SOR method
    :param h: The mesh spacing
    :param relaxation: The relaxation factor
    Returns a dictionary of results
    '''
    xCoord = 0.1-0.06 # Has equivalent potential to x = 0.06
    yCoord = 0.1-0.04 # Has the equivalent potentail to y = 0.04
    iCoord = int(xCoord // h)
    jCoord = int(yCoord // h)
    maxTries = 10000
    height = 0.02+0.08
    width = 0.06+0.04
    innerConductorVoltage = 110
    outerConductorVoltage = 0

    #Compute the n and m indexes
    n = int(width//h)
    m = int(height//h)
    corner = (int(0.04//h), int(0.02//h))

    potentialMatrix = np.zeros((n,m),dtype=np.float)
    #Populate the matrix with the boundary conditions
    for i in np.arange(n):
        for j in np.arange(m):
            if i <= corner[0] and j <= corner[1]:
                potentialMatrix[i,j] = innerConductorVoltage #Side f includes corner
            if i == n-1:
                potentialMatrix[i,j] = outerConductorVoltage #Side d
            if j == m-1:
                potentialMatrix[i,j] = outerConductorVoltage #Side b
            if j == corner[1] and i < corner[0]:
                potentialMatrix[i,j] = innerConductorVoltage #Side a excludes corner

    residualNorm = 1
    iterationNumber = 0
    tolerance = 10**(-5)
    previousGuess = np.copy(potentialMatrix)
    nextGuess = np.copy(potentialMatrix)
    allTolerable = False
    while(not allTolerable):
        for i in np.arange(n-1): #Do not include outer boundary
            for j in np.arange(m-1): #Do not include outer boundary
                if i <= corner[0] and j <= corner[1]:
                    pass #Do nothing as it is outside the problem domain
                elif i == 0:
                    if j > corner[1]:
                        #Handle Neuman boundary on c
                        nextGuess[i,j] = ( (1-relaxation)*previousGuess[i,j] + 
                                          (relaxation/4)*(2*previousGuess[i+1,j]+
                                                previousGuess[i,j+1] +
                                                nextGuess[i, j-1]) )
                elif j == 0: 
                    if i > corner[0]:
                        #Handle Neuman boundary on e
                        nextGuess[i,j] = ( (1-relaxation)*previousGuess[i,j] +
                                          (relaxation/4)*(previousGuess[i+1,j]+ 
                                                nextGuess[i-1,j] +
                                                2*previousGuess[i,j+1]) )
                else:
                    #Handle regular free nodes in the problem domain
                    nextGuess[i,j] = ( (1-relaxation)*previousGuess[i,j] +
                                            (relaxation/4)*(previousGuess[i+1,j]+ 
                                            nextGuess[i-1,j] +
                                            previousGuess[i,j+1] + 
                                            nextGuess[i, j-1]) )
        previousGuess = nextGuess
                     
        #compute residual
        allTolerable = True
        for i in np.arange(n-1): #Do not include outer boundary
            for j in np.arange(m-1):
                residual = 1
                if i <= corner[0] and j <= corner[1]:
                    residual = tolerance
                    pass
                elif i == 0:
                    if j >corner[1]:
                        residual = (2*nextGuess[i+1,j] + 
                                    nextGuess[i, j+1]+ 
                                    nextGuess[i, j-1]-
                                    4*nextGuess[i,j])
                elif j == 0:
                    if i >corner[0]:
                        residual = (nextGuess[i+1,j] + 
                                    nextGuess[i-1, j]+ 
                                    2*nextGuess[i, j+1]-
                                    4*nextGuess[i,j])
                else:
                    residual = (nextGuess[i+1,j] + 
                                        nextGuess[i-1, j]+ 
                                        nextGuess[i, j+1]+ 
                                        nextGuess[i, j-1]-
                                        4*nextGuess[i,j])
                if abs(residual) > tolerance:
                    allTolerable = False
        iterationNumber += 1
        if iterationNumber == maxTries:
            raise Exception(
                'You have exceeded the maximum number of iterations {}'.format(maxTries))
    
    result = {'h':h,'relaxation':relaxation,'iterations': iterationNumber, 
              'potentials': nextGuess, 'x,y':nextGuess[iCoord, jCoord]}
    return result

def finiteDifferenceJacobi(h):
    '''Solves for a set of potentials using finite difference approach.
    By exploiting symmetry, the following region is considered
    ----b--------
    |           |
    c           d
    |---a---|   |
            f   |
            |   |
            --e--
    where c and e are Neuman boundaries and the rest are dirchlet. Note, i, j = (0,0)
    starts in the lower left corner. Note, this formulation uses the Jacobi method.
    :param h: The size of the mesh spacing
    Returns a dictionary
    '''
    xCoord = 0.1-0.06 # Has equivalent potential to x = 0.06
    yCoord = 0.1-0.04 # Has the equivalent potentail to y = 0.04
    iCoord = int(xCoord // h)
    jCoord = int(yCoord // h)
    maxTries = 10000
    height = 0.02+0.08
    width = 0.06+0.04
    innerConductorVoltage = 110
    outerConductorVoltage = 0
    n = int(width//h)
    m = int(height//h)
    corner = (int(0.04//h), int(0.02//h))

    potentialMatrix = np.zeros((n,m),dtype=np.float)
    #Populate the matrix with the boundary conditions
    for i in np.arange(n):
        for j in np.arange(m):
            if i <= corner[0] and j <= corner[1]:
                potentialMatrix[i,j] = innerConductorVoltage #Side f includes corner
            if i == n-1:
                potentialMatrix[i,j] = outerConductorVoltage #Side d
            if j == m-1:
                potentialMatrix[i,j] = outerConductorVoltage #Side b
            if j == corner[1] and i < corner[0]:
                potentialMatrix[i,j] = innerConductorVoltage #Side a excludes corner

    residualNorm = 1
    iterationNumber = 0
    tolerance = 10**(-5)
    allTolerable = False
    while(not allTolerable):
        for i in np.arange(n-1): #Do not include outer boundary
            for j in np.arange(m-1): #Do not include outer boundary
                if i <= corner[0] and j <= corner[1]:
                    pass #Do nothing as it is outside the problem domain
                elif i == 0:
                    if j > corner[1]:
                        #Handle Neuman boundary on c
                        potentialMatrix[i,j] = ( (1/4)*(2*potentialMatrix[i+1,j]+
                                                potentialMatrix[i,j+1] +
                                                potentialMatrix[i, j-1]) )
                elif j == 0: 
                    if i > corner[0]:
                        #Handle Neuman boundary on e
                        potentialMatrix[i,j] = ( (1/4)*(potentialMatrix[i+1,j]+ 
                                                potentialMatrix[i-1,j] +
                                                2*potentialMatrix[i,j+1]) )
                else:
                    #Handle regular free nodes in the problem domain
                    potentialMatrix[i,j] = ( (1/4)*(potentialMatrix[i+1,j]+ 
                                            potentialMatrix[i-1,j] +
                                            potentialMatrix[i,j+1] + 
                                            potentialMatrix[i, j-1]) )
        #compute residual
        allTolerable = True
        for i in np.arange(n-1): #Do not include outer boundary
            for j in np.arange(m-1):
                residual = 1
                if i <= corner[0] and j <= corner[1]:
                    residual = tolerance
                    pass
                elif i == 0:
                    if j >corner[1]:
                        residual = (2*potentialMatrix[i+1,j] + 
                                    potentialMatrix[i, j+1]+ 
                                    potentialMatrix[i, j-1]-
                                    4*potentialMatrix[i,j])
                elif j == 0:
                    if i >corner[0]:
                        residual = (potentialMatrix[i+1,j] + 
                                    potentialMatrix[i-1, j]+ 
                                    2*potentialMatrix[i, j+1]-
                                    4*potentialMatrix[i,j])
                else:
                    residual = (potentialMatrix[i+1,j] + 
                                        potentialMatrix[i-1, j]+ 
                                        potentialMatrix[i, j+1]+ 
                                        potentialMatrix[i, j-1]-
                                        4*potentialMatrix[i,j])
                if abs(residual) > tolerance:
                    allTolerable = False
        iterationNumber += 1
        if iterationNumber == maxTries:
            raise Exception(
                'You have exceeded the maximum number of iterations {}'.format(maxTries))
    
    result = {'h':h,'iterations': iterationNumber, 
              'potentials': potentialMatrix, 'x,y':potentialMatrix[iCoord, jCoord]}
    return result

def relaxationTesting():
    '''Tests the relaxation of the finiteDifferencePotentialSolver
    Generates graphs and tabular output
    '''
    h = 0.02
    relaxations = np.linspace(1.1,2.0,num=10, endpoint=False) #Generate w
    iterations = []
    for relaxation in relaxations:
        result = finiteDifferencePotentialSolver(h, relaxation) #Compute finite diff
        #Tabluate the results
        print("Relaxation: {} | Iterations: {} | Phi(0.06,0.04): {}".format(
            truncateFloat(result['relaxation'],3),
            result['iterations'],
            truncateFloat(result['x,y'], 3)))

        iterations.append(result['iterations'])
    #Plot the resulting iterations and w
    plt.xlabel('Relaxation factor (w)')
    plt.ylabel('Number of iterations')
    plt.title('Number of iterations vs relaxation factor (w)')
    plt.grid(True)
    plt.plot(relaxations, iterations)
    plt.savefig('q3b.pdf', format='pdf')

def meshSizeTesting():
    '''Tests the finiteDifferencePotentialSolver and the finiteJacobi solver.
    Uses a fixed relaxation determined using relaxationTesting().
    Generates graphs and tabular output
    '''
    relaxation = 1.19 #Chosen based on the relaxation testing
    meshSizes = [0.02] #The starting mesh size
    outerConductor = 0
    innerConductor = 110
    x_y_SOR = []
    iterations_SOR = []
    x_y_Jacobi = []
    iterations_Jacobi = []
    meshInverse = []
    #Prepare the meshsizes
    for i in range(3): #Pick an upper limit that is reasonable given computation time
        nextMeshSize = meshSizes[len(meshSizes)-1] / 2
        meshSizes.append(nextMeshSize)
    
        
    for meshSize in meshSizes:
        meshInverse.append(1/meshSize)

        #Compute results for SOR
        result = finiteDifferencePotentialSolver(meshSize, relaxation)
        x_y_SOR.append(result['x,y'])
        iterations_SOR.append(result['iterations'])
        print('SOR Phi(0.04, 0.06): {} | 1/h {}'.format(truncateFloat(result['x,y'], 3),
                                                    1/meshSize))
        print('SOR Iterations: {} | 1/h {}'.format(result['iterations'],
                                                    1/meshSize))
        #Compute results for Jacobi
        result = finiteDifferenceJacobi(meshSize)
        x_y_Jacobi.append(result['x,y'])
        iterations_Jacobi.append(result['iterations'])
        print('Jacobi Phi(0.04, 0.06): {} | 1/h: {}'.format(
            truncateFloat(result['x,y'], 3), 1/meshSize))
        print('Jacobi Iterations: {} | 1/h: {}'.format(result['iterations'],
                                                    1/meshSize))
    #Graph of potential and inverse mesh
    plt.title('Potential Phi(0.04,0.06) versus 1/meshSize (h) using SOR')
    plt.xlabel('1/meshSize (h)')
    plt.ylabel('Potential Phi(0.04,0.06) (V)')
    plt.grid(True)
    plt.ylim([outerConductor, innerConductor]) #Display y over a meaningful range
    plt.plot(meshInverse, x_y_SOR, '-o')
    plt.savefig('sor_mesh_potential.pdf', format='pdf')
   
    #Graph of iterations and inverse mesh
    plt.clf()
    plt.cla()    
    plt.title('Number of iterations versus 1/meshSize (h) using SOR')
    plt.xlabel('1/meshSize (h)')
    plt.ylabel('Number of itereations')
    plt.grid(True)
    plt.ylim([0,iterations_SOR[len(iterations_SOR)-1]])
    plt.plot(meshInverse, iterations_SOR, '-o')
    plt.savefig('sor_mesh_iterations.pdf', format='pdf')
    
    
    #Graph of potential and inverse mesh
    plt.clf()
    plt.cla()
    plt.title('Potential Phi(0.04,0.06) versus 1/meshSize (h) using Jacobi')
    plt.xlabel('1/meshSize (h)')
    plt.ylabel('Potential Phi(0.04,0.06) (V)')
    plt.grid(True)
    plt.ylim([outerConductor, innerConductor]) #Display y over a meaningful range
    plt.plot(meshInverse, x_y_Jacobi, '-o')
    plt.savefig('jacobi_mesh_potential.pdf', format='pdf')
    
    #Graph of iterations and inverse mesh
    plt.clf()
    plt.cla()    
    plt.title('Number of iterations versus 1/meshSize (h) using Jacobi')
    plt.xlabel('1/meshSize (h)')
    plt.ylabel('Number of itereations')
    plt.grid(True)
    plt.ylim([0,iterations_Jacobi[len(iterations_Jacobi)-1]])
    plt.plot(meshInverse, iterations_Jacobi, '-o')
    plt.savefig('jacobi_mesh_iterations.pdf', format='pdf')
    
    return

if __name__ == '__main__':
    runLinearResistiveMeshTests()
    pass
