import numpy as np
from math import sqrt, factorial
import scipy.linalg as sl

"""
Implementacion de la clase que calcula la forma canonica de Jordan 
"""

class forma_jordan:
    
    def __init__(self, A):
        self.matrix = A
        self.nrow = np.shape(A)[0]
        self.ncol = np.shape(A)[1]
        self.eigenvalues, *_ = sl.eig(A) # eigenvalues de A
        self.jordan = np.zeros_like(A)
        self.change_base_matrix = np.zeros_like(A)
        self.bloques_jordan = np.array([])
        self.nilp = [] # indices de nilpotencia de los bloques de Jordan
        
    # Formato de Impresion
    def __repr__(self):
        return "{}".format(self.matrix)
    
    # Algoritmo de householder
    def householder_iterations(self, b, iterations):
            
        Qt = np.identity(self.nrow)
        R = 1*self.matrix

        for j in range(iterations):
            phi = sqrt(sum(R[j:self.nrow, j]**2))*np.sign(R[j,j])

            w = np.zeros((self.nrow , 1))
            w[j:self.nrow, 0] = R[j:self.nrow, j]
            w[j,0] += phi

            beta = 1/sum(w[j:self.nrow]**2)
            Hj = np.identity(self.nrow) - 2*beta*np.dot(w, w.T)
        
            R = np.dot(Hj, R)
            Qt = np.dot(Hj, Qt)
            b = np.dot(Hj, b)

        Q = Qt.T

        return R, Q, b 

    def householder(self, b):
        """
        Transforma el sistema Ax=b --> (Qt)Ax = (Qt)b, donde (Qt)A = R, Qt: transpuesta de Q
        Ademas Qt = Hn Hn-1 ... H2 H1, donde Hi es la i-esima matriz de householder, que elimina la i-esima columna
        """
        if self.nrow > self.ncol:
            R, Q, b = householder_iterations(self.matrix, b, self.ncol)
        else:
            R, Q, b = householder_iterations(self.matrix, b, self.nrow-1)

        return R, Q, b


    def gauss_jordan(self):
        """
        Calcula la inversa de a matriz A
        Retorna la inversa de A
        """
        if sl.det(self.matrix) != 0:
            invA = np.identity(self.nrow)
            ident = np.identity(self.nrow)
            A0 = 1*self.matrix
            for k in range(self.nrow):
                a = np.array(A0[:, k]/A0[k,k]).reshape(self.nrow, 1)
                a[k] = 1-1/A0[k,k]
                e =  np.zeros_like(a)
                e[k,0] = 1

                T = ident - np.dot(a, np.transpose(e))
                A0 = np.dot(T, A0)
                invA = np.dot(T, invA)
            return invA
        else:
            print("La matriz es singular, elija otro metodo.")



    def rango_matrix(self):
        """
        Calcula es rango de una matriz A
        """
        zeros = np.zeros((self.nrow, 1))
        A0, *_ = householder(self.matrix, zeros)
        # Calculo del rango de A0
        r = 0 # rango de A0
        for k in range(self.nrow):
            if not np.allclose(A0[k,:], 0.0):
                r += 1

        return r


    def power_matrix(self, k):
        """
        Calcula la potencia de A elevado a la k
        """
        A0 = np.identity(self.nrow)    
        for k in range(k):
            A0 = np.dot(A0, self.matrix)
        return A0


    # coef(i,k) = dr-i(k), r : rango de B0 = N(A-aI)^q
    # B0: matriz obtenida al aplicar las transformaciones de householder a (A-aI)^{q}
    def coeff(i, k, B0):
        r = rango_matrix(B0)

        if i == 0:
            return B0[r, k] # dr(k) = B0[r, k]
        else:
            # calculando di(k) = dr-ik 
            coeffik = B0[r-i,k]
            for j in range(i):
                coeffik -= (B0[r-i,r-i-j]/B0[r-i-j, r-i-j])*coeff(i+j, k, B0)

        return coeffik

    ############################################################################################################################
    ################################################    Convertir a metodos     ################################################
    ############################################################################################################################

    def busca_base(A, a, q):
        """
        Esta funcion calcula la base del nucleo N((A-aI)^{q})
        A: matriz
        a: valor propio de A
        q: menor indice tal que N((A-aI)^{q}) = N((A-aI)^{q+1})
        """
        
        
        nrowA = np.shape(A)[0]
        zeros = np.zeros((nrowA, 1))
        B0, *_ = householder(power_matrix(A-a*np.identity(nrowA), q), zeros)
        
        # calcula el rango de la matriz B0
        r = rango_matrix(B0)
        
        # G = [G1, G2, ..., Gn-r], donde Gi: i-esimo vector generador de N((A-aI)^{q})
        # NOTA: No esta garanizado que {G1, G2, ..., Gn-r}, sean li
        G = np.zeros((nrowA, nrowA-r))
        for k in range(r):
            # coef(i,k) = dr-i(k), r : rango de B0 = N(A-aI)^q
            G[k,:] = - np.array([coeff(r-k,r+i, B0) for i in range(1, nrowA-r+1)])/B0[k,k]
        
        G[r:, :] = np.identity(nrowA-r)
        
        return vectores_li(G, r) # retorna solo los vectore li de {G1, G2, ..., Gn-r}, en una matriz




    def vectores_li(G, r):
        """
        retorna solo los vectores li del conjunto {G1, G2, ..., Gn-r}, donde Gi = G[:,i] 
        y r = rango(B0) = dim(<N((A-aI)^{q})>)
        """ 
        # calcula en numero de vectores que son li de {G1, G2, ..., Gn-r}
        m = rango_matrix(G.T)
        if m == n-r:
            return G # todos los vectores  {G1, G2, ..., Gn-r} son li
        else:
            nrowG = np.shape(G)[0]
            M = G[:0].reshape(nrowG, 1)   # almacena los vectores li, por columna
            for k in range(1, n-r):
                v = G[:,k].reshape(nrowG, 1)
                if rango_matrix(M) < rango_matrix(np.concatenate(M, v, axis=1)):
                    M = np.concatenate(M, v, axis=1)

            return M


    # verifica si en espacio generado por Sq es igual al espacion generado por Sq1 
    # Como Sq y Sq1 son generada por la funcion "busca_base", esta son matrices y los vectores son sus columnas
    def equal_generate_space(Sq, Sq1):
        # Verifica si <Sq> esta incluido en <Sq1>, lo cual es suficiente para la igualdad porque dim(<Sq>) = dim(<Sq1>)
        dim = np.shape(Sq)[1] # dim: numero de comlumnas de Sq, es decir # de vectores 
        for k in range(dim):
            if rango_matrix(Sq1) > rango_matrix(np.concatenate(Sq1,Sq[:,k], axis=1)):
                return False
        return True


    def busca_vector(A, a, q):
        """
        Busca v tal que (A-aI)^{p}v = 0 and (A-aI)^{p-1}v != 0    
        """
        
        Sq = busca_base(A, a, q)
        dim = np.shape(Sq)[1] # calcula la dimension del espacio Sq
        nrowA = np.shape(A)[0] # numero de filas de A
        
        C0 = power_matrix(A-a*np.identity(nrowA), q-1) # C0 = (A-aI)^{q-1} 
        
        for k in range(dim):
                if not np.allclose(np.dot(C0,Sq[:,k]), 0.0):
                    return Sq[:,k]



    # implementar la funcion como un metodo de la clase "forma_jordan"
    def canonica_jordan(A):
        #spectro = eigenvalues(A)    # implementar calculo de eigenvalues de A
        base_bloques_jordan = np.array([])  # Base de la forma canónica de jordan
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
            base_bloque = np.zeros((q, q))
            for k in range(q-1,-1,-1):
                base_bloque[:,k] = v 
                v = np.dot(B, v)
            base_bloques_jordan.append(base_bloque)

            # Construye el bloque de jordan asociada a "base_bloque"
            BJ = np.zeros((q, q))
            for k in range(q):
                BJ[k,k] = a
                if k != q:
                    BJ[k, k+1] = 1

            jordan_bloques.append(BJ)
            nilp.append(q)
            base_bloques_jordan = np.append(base_bloques_jordan, base_bloque)


        # Hallando Forma canonica de jordan(J)
        nrowJ = sum(nilp) # calcula el orden de la matriz de jordan
        JA = np.zeros((nrowJ, nrowJ))
        q0 = 0

        for k, q in enumerate(nilp):
            JA[q0:q, q0:q] = jordan_bloques[k]
            q0 = q

        """
        NOTA: ver como acceder a la base (que esta en matrices)

        #Hallando la matriz de cambio de base (P)
        invP = np.zeros((nrowJ, nrowJ))
        for k in range(nrowJ):
            invP[:,k] = gamma(k)

        """
        
        # calculando P , inversa de invP
        #P = gauss_jordan(invP)

        return JA, P, jordan_bloques, nilp


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

        """
        JA, P, jordan_bloques, nilp = canonica_jordan(A)
        nrowA = np.shape(A)[0]
        JAK = np.zeros_like(A)  # Potencia k-esima de la forma canonica de jordan 
        m0 = 0 # inicializacion de indexador
        m = 0   # indice para indexar los bloques de jordan

        for q, a in zip(nilp, eigenvalues(A)) :    ################### Implementar calculo de eigenvalues ##########################
            F = np.array([factorial(k)/(factorial(j)*factorial(k-j))*a**k for j in range(q)])
            BJK = np.zeros((q,q))
            for j in range(q):
                BJK[j, j:] = F[1:q-j+1]
            m += q
            JAK[m0:m, m0:m] = BJK
            m0 = m

        invP = gauss_jordan(P)
        invP_JK = np.dot(invP, JAK)
        AK = np.dot(invP_JK, P)

        return AK
