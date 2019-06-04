import numpy as np
import numpy as np
from math import sqrt


def householder(A, b):
    """
    Transforma el sistema Ax=b --> (Qt)Ax = (Qt)b, donde (Qt)A = R, Qt: transpuesta de Q
    Ademas Qt = Hn Hn-1 ... H2 H1, donde Hi es la i-esima matriz de householder, que elimina la i-esima columna
    """
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


def rango_matrix(A):
    """
    Calcula es rango de una matriz A
    """
    nrowA = np.shape(A)[0]
    zeros = np.zeros((nrowA, 1))
    A0, *_ = householder(A, zeros)
    # Calculo del rango de A0
    r = 0 # rango de A0
    for k in range(nrowA):
        if not np.allclose(A0[k,:], 0.0):
            r += 1

    return r


def comparar(A,v):  #devuelve True si son l.i
    validar= False
    
    if A.shape[1]== v.shape[1]:   #comprueba si la dimension de las columnas son iguales
     
        if A.shape[0]==1:

            rango1=1
            M1 = np.append(A,v,axis=0) 
           
            rango2=rango_matrix(M1)
           
        else:
            
            rango1=rango_matrix(A)

            M1=np.append(A,v,axis=0)  #Agrega v al espacio fila de A

            rango2=rango_matrix(M1)
       
        if rango1<rango2:
            validar= True #los vectores fila de A y v son li
        return validar 
       
            
    else:
        print("La dimension de las columnas no son iguales")
        #return -1    opcional



def vectores_li(G):
    # calcula en numero de vectores que son li de {G1, G2, ..., Gn-r},  Gi columnas
 
     
    m = rango_matrix(np.transpose(G))

    n=G.shape[1]  
    if m == n: #rango de R1 comparado con nº de columnas de G
        return G # todos los vectores  {G1, G2, ..., Gn-r} son li
    else:
        M=np.transpose(G)
        
        li = M[0,:] #primera fila de la transpuesta
        li=np.array([li])
    
        for k in range(1, n):
           
            R=M[k,:]
            R=np.array([R])
            
            
            if comparar(li,R):
                li=np.append(li, R,axis=0)
                
        li=np.transpose(li)
        return li


A=np.array([[1, 1, 3],
            [1, 7, 3],
            [1, 3, 3] ])
print(vectores_li(A))

