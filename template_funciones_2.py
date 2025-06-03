import numpy as np

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

#%%
def construye_adyacencia(D,m):
    #copia D
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexa vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # elege todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convierte a entero
    np.fill_diagonal(A,0) # Borra diagonal para eliminar autolinks
    return(A)

def calcula_matriz_K(A):
    # n dimensión de A
    n = A.shape[0]
    #Genera una matriz cuadrada K de nxn con todos 0
    K = np.zeros((n, n))
    for i in range(n):
        #Suma los valores de cada fila de A.
        valor_fila = A[i].sum()
        #Agrega en la posición correspondiente de la diagonal de K valor_fila.
        K[i, i] = valor_fila
    #retorna K
    return K

def calcula_L(A):
    K = calcula_matriz_K(A)
    L = K - A
    return L

def calcula_P(A):
    grados = np.sum(A, axis=1)
    m = np.sum(A) / 2 
    n = A.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = (grados[i] * grados[j]) / (2 * m)
    return P


def calcula_R(A):
    P = calcula_P(A)
    return A - P


#%%
def calcula_s(v):
    aux = []
    for coordenada in v:
        if coordenada >= 0:
            aux.append(1)
        else:
            aux.append(-1)
    s = np.array(aux).reshape(-1, 1)
    return s
def calcula_lambda(L,v):
    s = calcula_s(v)
    lambdaa = (1/4) * (s.T @ L @ s)
    return lambdaa

def calcula_Q(R,v):
    s = calcula_s(v)
    Q = s.T @ R @ s
    return Q

#%%
def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   A = np.atleast_2d(A)  # Asegura que A sea 2D
   n, m = A.shape

   v = np.random.rand(n) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v/ np.linalg.norm(v) # Lo normalizamos
   
   v1 = A @ v  # Aplicamos la matriz una vez
   v1 = (v1)/np.linalg.norm(v1) # normalizamos
   l = (v1.T @ A @ v1) / (v1.T @ v1) # Calculamos el autovalor estimado
   nrep = 0 # Contador
   
   l1 = (v1.T @ A @ v1) / (v1.T @ v1) # Y el estimado en el siguiente paso   
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v1 # Calculo nuevo v1
      v1 = (v1)/np.linalg.norm(v1) # Normalizo
      l1 = (v1.T @ A @ v1) / (v1.T @ v1) # Calculo autovalor
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   return v1,l,nrep<maxrep
#%%
def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1, _ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

#%%
def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = A - l1 * np.outer(v1, v1)
   return metpot1(deflA,tol,maxrep)

#%%
def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    #calculo el autovalor la inversa de A + mu * I
    n,_ = A.shape #tomo la dimension de A
    I = np.eye(n) #creo la matriz identidad de dimension n
    X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
    iX = np.linalg.inv(X)
    return metpot1(iX,tol=tol,maxrep=maxrep)

#%%
def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   n,_ = A.shape #tomo la dimension de A
   I = np.eye(n) #creo la matriz identidad de dimension n
   X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
   iX = np.linalg.inv(X) # La invertimos
   defliX = deflaciona(iX,tol=1e-8,maxrep=np.inf) # La deflacionamos
   v,l,_ = metpot1(defliX,tol,maxrep)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v1,l1,_ = metpot1(A,tol=1e-8,maxrep=np.inf)
        v,_,_ = metpot2(L,v1,l1)# Encontramos el segundo autovector de L
        
        # Separamos los nodos en dos grupos según el signo del segundo autovector
        ind_p = []
        ind_m = []
        umbral = np.median(v)
        for i in range(len(v)):
            if v[i] >= umbral:
                ind_p.append(i)
            else:
                ind_m.append(i)

                
        if len(ind_p) == 0 or len(ind_m) == 0:
            return [list(nombres_s)]

        # Creamos las submatrices A_p y A_m
        Ap = np.array([[A[i][j] for j in ind_p] for i in ind_p])
        Am = np.array([[A[i][j] for j in ind_m] for i in ind_m])
        
        # Obtenemos los nombres de los nodos correspondientes a cada grupo
        nombres_p = [nombres_s[i] for i in ind_p]
        nombres_m = [nombres_s[i] for i in ind_m]

        return laplaciano_iterativo(Ap, niveles - 1, nombres_p) + \
              laplaciano_iterativo(Am, niveles - 1, nombres_m)
#%%

def modularidad_iterativo(A=None, R=None, nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [nombres_s]
    else:
        v, l, _ = metpot1(R)  # Primer autovector y autovalor de R
        print("Primer autovector:", v)
        # Modularidad Actual:
        Q0 = np.sum(R[v > 0, :][:, v > 0]) + np.sum(R[v < 0, :][:, v < 0])
        print("Modularidad Q0:", Q0)
        if Q0 <= 0 or all(v > 0) or all(v < 0):  # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [nombres_s]
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            indices_pos = [i for i in range(len(v)) if v[i] > 0]
            indices_neg = [i for i in range(len(v)) if v[i] < 0]
            Rp = R[np.ix_(indices_pos, indices_pos)]  # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(indices_neg, indices_neg)]  # Parte asociada a los valores negativos de v
            vp, lp, _ = metpot1(Rp)  # autovector principal de Rp
            
            vm, lm, _ = metpot1(Rm)  # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp > 0) or all(vp < 0):
                Q1 = np.sum(Rp[vp > 0, :][:, vp > 0]) + np.sum(Rp[vp < 0, :][:, vp < 0])
            if not all(vm > 0) or all(vm < 0):
                Q1 += np.sum(Rm[vm > 0, :][:, vm > 0]) + np.sum(Rm[vm < 0, :][:, vm < 0])
            if Q0 >= Q1:  # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
    
                return [[ni for ni, vi in zip(nombres_s, v) if vi > 0],
                        [ni for ni, vi in zip(nombres_s, v) if vi < 0]]
            else:
                # Sino, repetimos para los subniveles
                comunidad1 = modularidad_iterativo(None, Rp, [nombres_s[i] for i in indices_pos])
                comunidad2 = modularidad_iterativo(None, Rm, [nombres_s[i] for i in indices_neg])
                return comunidad1 + comunidad2
#%% pruebaç
#print(laplaciano_iterativo(A_ejemplo,2))
print(modularidad_iterativo(A_ejemplo))
