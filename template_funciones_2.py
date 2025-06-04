import numpy as np
from scipy.linalg import solve_triangular
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
""" FUNCIONES DEL TP1 """
def calculaLU(A):
    m=A.shape[0] # cantidad de filas de A
    n=A.shape[1] # cantidad de columnas de A
    #copia A
    Ac = A.copy()
    #si la matriz no es cuadrada termina la función
    if m!=n:
        print('Matriz no cuadrada')
        return
    #recorre las columnas de la matriz
    for j in range (n):
        #recorre las filas debajo de la diagonal
        for i in range(j+1,n):
            # Calcula el multiplicador de eliminación
            Ac[i,j] = Ac[i,j] / Ac[j,j]
            for k in range(j + 1, n):
                # Resta el múltiplo correspondiente de la fila j a la fila i
                Ac[i, k] = Ac[i, k] - Ac[i, j] * Ac[j, k]
    L = np.tril(Ac,-1) + np.eye(m) #toma la parte inferior de Ac sin la diagonal y le suma la identidad para que la diagonal de L sea todo 1
    U = np.triu(Ac) #toma la parte superior de Ac incluyendo la diagonal
    return L, U

# Calcula la inversa de una matriz cuadrada A utilizando su descomposición LU.
# Recibe: A (matriz).
# Retorna: A_inv (matriz): matriz inversa de A.

def calcula_matriz_inversa(A):
    # n dimensión de A
    n = A.shape[0]
    L, U = calculaLU(A) #Descompone A en L (lower) y U (upper
    #matriz Identidad de nxn
    I = np.eye(n)
    #matriz de todos 0
    A_inv = np.zeros_like(A)

    for i in range(n):
        e_i = I[:, i] # Toma una columna de la identidad
        y = solve_triangular(L, e_i, lower=True)    # Resuelve Ly = e_i
        x_i = solve_triangular(U, y, lower=False)   # Resuelve Ux = y
        A_inv[:, i] = x_i # Guarda esa columna en la inversa
    #retorna A_inv
    return A_inv

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

#%%
def calcula_L(A):
    K = calcula_matriz_K(A)
    L = K - A
    return L
L = calcula_L(A_ejemplo)

#%%
def calcula_P(A):
    conexiones = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            conexiones += A[i][j]
    P = np.zeros((n, n))
    K = calcula_matriz_K(A)
    for i in range(n):
        P[i, i] = (K[i, i] * K[i, i]) / conexiones
    return P

P = calcula_P(A_ejemplo)
#%%
def calcula_R(A):
    P = calcula_P(A)
    R = A - P
    return R

R = calcula_R(A_ejemplo)

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
#solo es para el ejemplo
#Calculamos corte y modularidad para la A_ejemplo
# Calculamos autovalores y autovectores
valores, vectores = np.linalg.eig(A_ejemplo)

# Tomamos el autovalor más grande y su correspondiente autovector
indice_max = np.argmax(valores)
autovalor_principal = valores[indice_max]
autovector_principal = vectores[:, indice_max]

lambda_ejemplo = calcula_lambda(L, autovector_principal)

q_ejemplo = calcula_Q(R, autovector_principal)
#%%
def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   n,_ = A.shape
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

autovalor_max_ejemplo = metpot1(A_ejemplo)
#%%
def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1, _ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

deflaciona_ejemplo = deflaciona(A_ejemplo)

#%%
def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = A - l1 * np.outer(v1, v1)
   return metpot1(deflA,tol,maxrep)
v1,l1,_ = autovalor_max_ejemplo
segundo_autovalor_ejemplo = metpot2(A_ejemplo, v1, l1)
#%%
def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    #calculo el autovalor la inversa de A + mu * I
    n,_ = A.shape #tomo la dimension de A
    I = np.eye(n) #creo la matriz identidad de dimension n
    X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
    iX = calcula_matriz_inversa(X)
    return metpot1(iX,tol=tol,maxrep=maxrep)
#%%
def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   n,_ = A.shape #tomo la dimension de A
   I = np.eye(n) #creo la matriz identidad de dimension n
   X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
   iX = calcula_matriz_inversa(X) # La invertimos
   defliX = deflaciona(iX,tol=1e-8,maxrep=np.inf) # La deflacionamos
   v,l,_ = metpot1(defliX,tol,maxrep)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_
#%%

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
        for i in range(len(v)):
            if v[i] >= 0:
                ind_p.append(i)
            if v[i] < 0:
                ind_m.append(i)
                
        if len(ind_p) == 0 or len(ind_m) == 0:
            return [list(nombres_s)]

        # Creamos las submatrices A_p y A_m
        Ap = A[np.ix_(ind_p, ind_p)]
        Am = A[np.ix_(ind_m, ind_m)]
        
        # Obtenemos los nombres de los nodos correspondientes a cada grupo
        nombres_p = [nombres_s[i] for i in ind_p]
        nombres_m = [nombres_s[i] for i in ind_m]

        return laplaciano_iterativo(Ap, niveles - 1, nombres_p) + \
              laplaciano_iterativo(Am, niveles - 1, nombres_m)
#%%
def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return np.nan
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = list(range(R.shape[0]))
    else:
        nombres_s = list(nombres_s)
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [list(nombres_s)]
    
    v,l,_ = metpot1(R) # Primer autovector y autovalor de R
    
    # Dividir según signo
    ind_p = [i for i in range(len(v)) if v[i] >= 0]
    ind_m = [i for i in range(len(v)) if v[i] < 0]
    
    if len(ind_p) == 0 or len(ind_m) == 0:
        return [list(nombres_s)]
    
    #Modularidad actual
    Q0 = np.sum(R[np.ix_(ind_p, ind_p)]) + np.sum(R[np.ix_(ind_m, ind_m)])

    # Creamos las submatrices A_p y A_m
    Rp = R[np.ix_(ind_p, ind_p)] # Parte de R asociada a los valores positivos de v
    Rm = R[np.ix_(ind_m, ind_m)] # Parte asociada a los valores negativos de v
            
    vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
    vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
    # Calculamos el cambio en Q que se produciría al hacer esta partición
    Q1 = 0
    
    if not (all(vp >= 0) or all(vp < 0)):
        ind_vp_p = [i for i in range(len(vp)) if vp[i] >= 0]
        ind_vp_m = [i for i in range(len(vp)) if vp[i] < 0]
        Q1 += np.sum(Rp[np.ix_(ind_vp_p, ind_vp_p)]) + np.sum(Rp[np.ix_(ind_vp_m, ind_vp_m)])

    if not (all(vm >= 0) or all(vm < 0)):
        ind_vm_p = [i for i in range(len(vm)) if vm[i] >= 0]
        ind_vm_m = [i for i in range(len(vm)) if vm[i] < 0]
        Q1 += np.sum(Rm[np.ix_(ind_vm_p, ind_vm_p)]) + np.sum(Rm[np.ix_(ind_vm_m, ind_vm_m)])

    if Q0 >= Q1:
        return [[nombres_s[i] for i in ind_p], [nombres_s[i] for i in ind_m]]
    else:
        nombres_p = [nombres_s[i] for i in ind_p]
        nombres_m = [nombres_s[i] for i in ind_m]
        return modularidad_iterativo(R=Rp, nombres_s=nombres_p) + modularidad_iterativo(R=Rm, nombres_s=nombres_m)
#%% pruebaç
#print(laplaciano_iterativo(A_ejemplo,2))
print(modularidad_iterativo(A_ejemplo))

#%%
#%%
lista_m = [3,5,10,50]
cantidad_de_grupos = modularidad_iterativo(A_ejemplo)
niveles = cantidad_de_grupos

def comunidades(D,lista ,int):
    for i in lista:
        A = construye_adyacencia(D, i)
       #A_prima = (1/2*(A + A.T)) innecesario porque A+A.T = 2A entonces 1/2*2A = A
        if int == 0:
           laplaciano_iterativo(A, niveles)
        else:
            modularidad_iterativo(A)
        #HACER GRAFICOS
    #return grafico
