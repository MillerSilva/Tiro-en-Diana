import numpy as np
from math import sqrt, factorial
import scipy.linalg as sl


"""
def householder(A, b):
    
    #Transforma el sistema Ax=b --> (Qt)Ax = (Qt)b, donde (Qt)A = R, Qt: transpuesta de Q
    #Ademas Qt = Hn Hn-1 ... H2 H1, donde Hi es la i-esima matriz de householder, que elimina la i-esima columna
    
    nrowA = np.shape(A)[0]
    Qt = np.identity(nrowA)
    R = 1*A
    for j in range(nrowA):
        phi = sqrt(sum(R[j:nrowA, j]**2))*np.sign(R[j,j])

        w = np.zeros((nrowA, 1))
        w[j:nrowA, 0] = R[j:nrowA, j]
        w[j,0] += phi

        beta = 1/sum(w[j:nrowA]**2)
        Hj = np.identity(nrowA) - 2*beta*np.dot(w, w.T)

        R = np.dot(Hj, R)
        Qt = np.dot(Hj, Qt)
        b = np.dot(Hj, b)
    Q = Qt.T

    return R, Q, b

"""

"""
algoritmo probado para matrices cuadradas y no cuadradas
"""


# probado(falta implementar en clase)
def householder_iterations(A, b, iterations):
    
    nrowA, ncolA = np.shape(A)
    Qt = np.identity(nrowA)
    R = 1*A

    for j in range(iterations):
        phi = sqrt(sum(R[j:nrowA, j]**2))*np.sign(R[j,j])

        w = np.zeros((nrowA, 1))
        w[j:nrowA, 0] = R[j:nrowA, j]
        w[j,0] += phi

        beta = 1/sum(w[j:nrowA]**2)
        Hj = np.identity(nrowA) - 2*beta*np.dot(w, w.T)
    
        R = np.dot(Hj, R)
        Qt = np.dot(Hj, Qt)
        b = np.dot(Hj, b)

    Q = Qt.T

    return R, Q, b

# probado(falta implementar en clase)
def householder(A, b):
    """
    Transforma el sistema Ax=b --> (Qt)Ax = (Qt)b, donde (Qt)A = R, Qt: transpuesta de Q
    Ademas Qt = Hn Hn-1 ... H2 H1, donde Hi es la i-esima matriz de householder, que elimina la i-esima columna
    """
    nrowA, ncolA = np.shape(A)
    if nrowA > ncolA:
        R, Q, b = householder_iterations(A, b, ncolA)
    else:
        R, Q, b = householder_iterations(A, b, nrowA-1)

    return R, Q, b
        
        
        
    