"""
Algoritmo que calcula la forma canonica de Jordan de una matrix
"""
import numpy as np
import scipy.linalg as sl
from math import sqrt


################################################################################################################################
############################################ FUNCIONES PROBADAS ################################################################
################################################################################################################################


# probado(falta implementar en clase)
def householder_iterations(A, b, iterations):
    """
    Transforma el sistema Ax=b --> (Qt)Ax = (Qt)b, donde (Qt)A = R, Qt: transpuesta de Q
    Ademas Qt = Hn Hn-1 ... H2 H1, donde Hi es la i-esima matriz de householder, que elimina la i-esima columna
    """
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


    
# probado(falta implementar en clase)
def gauss_jordan(A):
    """
    Calcula la inversa de a matriz A
    Retorna la inversa de A
    """
    if sl.det(A) != 0:
        nrowA = np.shape(A)[0]
        invA = np.identity(nrowA)
        ident = np.identity(nrowA)
        for k in range(nrowA):
            a = np.array(A[:, k]/A[k,k]).reshape(nrowA, 1)
            a[k] = 1-1/A[k,k]
            e =  np.zeros_like(a)
            e[k,0] = 1

            T = ident - np.dot(a, np.transpose(e))
            A = np.dot(T, A)
            invA = np.dot(T, invA)
        return invA        
    else:
        print("La matriz es singular, elija otro metodo.")


# probado(falta implementar en clase)
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


# probado(falta implementar en clase)
def power_matrix(A, k):
    """
    Calcula la potencia de A elevado a la k
    """
    nrow = np.shape(A)[0]
    A0 = np.identity(nrow)    
    for k in range(q):
        A0 = np.dot(A0, A)
    
    return A0



###################################################################################################################################
############################################ FUNCIONES NO PROBADAS ################################################################
###################################################################################################################################

"""
B0x = 0, x = [x1, x2, ..., xr, xr+1, ..., xn], como r = rango(B0), los xi, i=1-r, pueden colocarse como
combinacion lineal de xr+1, xr+2, ..., xn. Asi 

-B0[i,i]xi = dr+1(k)xr+1 + dr+2(k)xr+2 + ... + dn(k)xn

coeff(k, i)= di(k): coeficiente de xi, i=r+1,..,n en la combinacion lineal de los xr+1,xr+2,...,xn, para obtener 
-B0[i,i]xi
"""
#Definido en la clase, pero falta su transformacion a un metodo
def coeff(k, i, B0):
    r = rango_matrix(B0)
    j = r-i
    if j == 0:
        return B0[r, k] # dr(k) = B0[r, k]
    else:
        # calculando di(k) = dik
        dik = B0[i,k]
        for k in range(j):
            dik -= (B0[i,i-k]/B0[i-k, i-k])*coeff(k, i-k, B0)

    return dik


"""
retorna solo los vectores li del conjunto {G1, G2, ..., Gn-r}, donde Gi = G[:,i] 
y r = rango(B0) = dim(<N((A-aI)^{q})>)
"""  
# def vectores_li(G, r):
#     # calcula en numero de vectores que son li de {G1, G2, ..., Gn-r}
#     m = rango_matrix(G.T)
#     if m == n-r:
#         return G # todos los vectores  {G1, G2, ..., Gn-r} son li
#     else:
#         li = np.array([G[:,0]])
#         for k in range(1, n-r):
#             ################### implementar v\in <li> #######################################
#             if G[:, k] in <li>:
#                 np.append(li, G[:,k])

        ####################### retornar li en una matriz ###################################3

    

def busca_base(A, a, q):
    """
    Esta funcion calcula la base del nucleo N((A-aI)^{q})
    A: matriz
    a: valor propio de A
    q: menor indice tal que N((A-aI)^{q}) = N((A-aI)^{q+1})
    """
    
    
    nrow = np.shape(A)[0]
    zeros = np.zeros((nrow, 1))
    B0, _ = householder(power(A-a*np.identity(nrow), q), zeros)
    
    # calcula el rango de la matriz B0
    r = rango_matrix(B0)
    
    # G = [G1, G2, ..., Gn-r], donde Gi: i-esimo vector generador de N((A-aI)^{q})
    # NOTA: No esta garanizado que {G1, G2, ..., Gn-r}, sean li
    G = np.zeros((nrow, nrow-r))
    for k in range(r):
        # coeff(k, i): coeficiente de xi en la combinacion lineal de los xr+1,xr+2,...,xn, para obtener -B0[k,k]xk
        G[k,:] = - np.array([coeff(k,r+i, B0) for i in range(1, nrow-r+1)])/B0[k,k]
    G[nrow:, :] = np.identity(nrow-r)
    
    return vectores_li(G, r) # retorna solo los vectore li de {G1, G2, ..., Gn-r}, en una matriz


# verifica si en espacio generado por Sq es igual al espacion generado por Sq1 
# Como Sq y Sq1 son generada por la funcion "busca_base", esta son matricez y los vectores son sus columnas
def equal_generate_space(Sq, Sq1):
    # Verifica si <Sq> esta incluido en <Sq1>, lo cual es suficiente para la igualdad porque dim(<Sq>) = dim(<Sq1>)
    dim = np.shape(Sq)[1] # dim: numero de comlumnas de Sq, es decir # de vectores 
    for k in range(dim):
        # nota: A.T == np.transpose(A)
        if rango_matrix(Sq1) > rango_matrix(np.concatenate(sq1,Sq[:,k].T), axis=1):
            return False
    return True


def busca_vector(A, a, p):
    """
    Busca v tal que (A-aI)^{p}v = 0 and (A-aI)^{p-1}v != 0    
    """
     
    Sq = busca_base(A, a, q)
    dim = np.shape(Sq)[1] # calcula la dimension del espacio Sq
    nrow = np.shape(A)[0] # numero de filas de A
    
    C0 = power(A-a*np.identity(nrow), q-1) # C0 = (A-aI)^{p-1} 
    C1 = np.dot(C0, A-a*np.identity(nrow)) # C1 = (A-aI)^{p}
    
    for k in range(dim):
            if np.allclose(np.dot(B1,Sq[:,k]), 0.0) and not np.allclose(np.dot(B0,Sq[:,k]), 0.0):
                return Sq[:,k]
    

# implementar la funcion como un metodo de la clase "forma_jordan"
def canonica_jordan(A):
    spectro = eigenvalues(A)
    gamma = np.array([]) # Base de la forma canónica de jordan
    nilp = [] # indices de nilpotencia de los espacios
    jordan_bloques = np.array([]) # almacena los bloques de jordan

    for a in spectro:
        q = 1
        Sq = busca_base(A, a, q)
        Sq1 = busca_base(A, a, q+1)

        while equal_generate_space(Sq, Sq1):
            q += 1
            Sq = Sq1
            Sq1 = busca_base(A, a, q+1)


        v = busca_vector(A, a, q)

        # Construye la base que genera un bloque de jordan
        nrowA = np.shape(A)[0]
        B = A - a*np.identity(nrowA)
        base_bloque = [v]
        for k in range(q-1):
            base_bloque.append(np.dot(B, v))
            v = np.dot(B, v)


        # Construye el bloque de jordan asociada a "base_bloque"
        J = np.zeros((q, q))
        for k in range(q):
            J[k,k] = a
            if k != q:
                J[k, k+1] = 1

        jordan_bloques.append(J)
        nilp.append(q)
        gamma = np.append(gamma, base_bloque.sort(reverse=1))


    # Hallando Forma canonica de jordan
    nrowJ = sum(nilp) # calcula el orden de la matriz de jordan
    JA = np.zeros((nrowJ, nrowJ))
    q0 = 0

    for k, q in enumerate(nilp):
        JA[q0:q, q0:q] = jordan_bloques(k)
        q0 = q

    #Hallando la matriz de cambio de base (P)
    invP = np.zeros((nrowJ, nrowJ))
    for k in range(nrowJ):
        invP[:,k] = gamma(k)

    # calculando P , inversa de invP
    P = gauss_jordan(invP)

    return JA, P, jordan_bloques, nilp
    
     
# implementado en la clase
def power_jordan(A, k):
    
    """
        Calcula la potencia k-esima de A, mediante la matriz de jordan.
        NOTA: Al calcular la potencia de la matriz de jordan, lo 
        JK: matriz de jordan elevada a la k
        BJ: Bloque de jordan
        BJK_ Bloque de jordan elevado a la k
        PJK: P * JK
        AK: matriz A elevada a la potencia k
        invP: inversa de P
        invPJK: invP * JK

        PROBLEM:
        a: valor propio del bloque de jordan(no esta disponible)
    """
    JA, P, jordan_bloques, nilp = canonica_jordan(A)
    nrowA = np.shape(A)[0]
    JK = np.zeros_like(A)
    m0 = 0 # inicializacion de indexador

    for m, BJ in zip(nilp, jordan_bloques):
        F = np.array([factorial(k)/(factorial(j)*factorial(k-j))*a**k for j in range(m)])
        BJK = np.zeros_like(m)
        for j in range(m):
            BJK[j, j:] = F[1:m-j+1]
        JK[m0:m, m0:m] = BJK
        m0 = m

    invP = gauss_jordan(P)
    invPJK = np.dot(invP, JK)
    AK = np.dot(invPJK, P)

    return AK
            
                                                
                                                
##################################################################################################################################
######################################## FUNCIONES A IMPLEMENTAR #################################################################
##################################################################################################################################
# NOTA: Por el monento vamos a usar la funcion predefinida(scipy.linalg.eig), para calcular los valores propio

                                                
"""

    #NOTA: Falta combinar el uso del metodo potencia con deflaccion de la matriz



def M_Potencias(A,X, N = 100):
        
    n=A.size
    p=0
    for i in range(n):
        if X[i]==sl.norm(X, np.inf):
            p=i
            break
    X=np.dot(X, 1.0/X[p])
    for k in range(1,N):
        Y=[0]*n
        Y=np.dot(A,X)
        u=Y[p]
        m=0
        for i in range(n):
            if Y[i]==sl.norm(Y, np.inf):
                m=i
                break
        if Y[m]==0:
            print 'A tiene el valor caracteristico 0, selecione un nuevo vector x y reinicie'
            break
        X=np.dot(Y, 1.0/Y[p])
    return u,X

# Implementa la deflaccion de la matriz A 
def deflation(A):
    a,x = M_Potencias(A, X0) # codificar, guardar valores propios; X0: inicializacion del vector para el metodo potencia
    n = x.size
    if x[0] != 0:
        phi = np.sign(x[0])*sl.norm(x)
    else:
        phi = sl.norm(x)
    e = np.zeros((n, 1))
    e[0] = 1
    w = x + phi*e
    T = np.identity(n) - (2.0 / sl.norm(w)**2) * np.dot(w, w.T)
    D = T.T * A * T
    C = D[1:n, 1:n]

    return C

"""

def M_eigvals_temp(A):
	ev = sl.eigval(A)
	img = np.imag(ev)
	if np.allclose(img, 0.0) == True:
		return np.real(ev)
	else:
		return ev
		
