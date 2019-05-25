import numpy as np


def conteo_filas0(H):  #usar  una vez reducido
    V=np.zeros((1,H.shape[1]))
    r=0
    c=0
    B=H
    while c<B.shape[0]:
        if not np.array_equiv(B[c,:],V[0,:]) :  
            r+=1
        c+=1
    return r
    
    ########################

def comparar(A,v):  #devuelve True si son li
    validar= False
    
    if A.shape[1]== v.shape[1]:   #comprueba si la dimension de las columnas son iguales
        if A.shape[0]==1:
            rango1=1
           
            M =np.append(A,v,axis=0) 
            R1 = np.linalg.qr(M,'r')
            R1=R1.round(8)
            rango2=conteo_filas0(R1)
    
        else:
            
            R1 = np.linalg.qr(A,'r')
            R1=R1.round(8)
            rango1=conteo_filas0(R1)

            

            M=np.append(A,v,axis=0)  #Agrega v al espacio fila de A
            R1 = np.linalg.qr(M,'r')
            R1=R1.round(8)
            rango2=conteo_filas0(R1)
       
        if rango1<rango2:
            validar= True #los vectores fila de A y v son li
        return validar    
            
    else:
        print("La dimension de las columnas no son iguales")
        #return -1    opcional



def vectores_li(G, r):
    # calcula en numero de vectores que son li de {G1, G2, ..., Gn-r},  Gi columnas
    R1 = np.linalg.qr(np.transpose(G),'r')  
    R1=R1.round(8)
    m = conteo_filas0(R1)

    n=G.shape[1]
 
    if m == n-r:
        return G # todos los vectores  {G1, G2, ..., Gn-r} son li
    else:
        M=np.transpose(G)
        
        li = M[0,:] #primera fila de la transpuesta
        li=np.array([li])
        for k in range(1, n-r):
           
            R=M[k,:]
            R=np.array([R])
            
            if comparar(li,R):
                li=np.append(li, R,axis=0)
                
        li=np.transpose(li)
        return li

       

 


#prueba
A = np.array([[1,2,3,4],[3,5,4,5],[2,4,6,8],[2,7,6,8]])
A=np.transpose(A)

print(vectores_li(A,2))

