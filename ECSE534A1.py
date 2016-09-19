import numpy as np
import random

def createTestMatrix(size):
    #add size check
    testMatrix = None
    aMatrix = np.zeros((size,size))
    for row in np.nditer(aMatrix, op_flags=['readwrite']):
        row[...] = random.randrange(1, 9)
    
    bMatrix = np.transpose(aMatrix)
    testMatrix = aMatrix.dot(bMatrix)
    return testMatrix

if __name__ == '__main__':
    matrix = createTestMatrix(2)
    print(matrix)