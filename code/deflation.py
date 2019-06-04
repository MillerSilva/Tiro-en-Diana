import numpy as np
import scipy.linalg as sl

def M_Potencias(A,X):
        N=input('([Met_Potencias] Ingrese numero de iteraciones:')
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
                print ('A tiene el valor caracteristico 0, selecione un nuevo vector x y reinicie')
                break
            X=np.dot(Y, 1.0/Y[p])
        return u,X

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
    T = np.identity(n) - (2.0 / sl.norm(w)) * np.dot(w, w.T)
    C = T.T * A * T
    C = C[1:n, 1:n]

