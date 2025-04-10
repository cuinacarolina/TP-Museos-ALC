# TP-Museos-ALC
def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    # Completar! Have fun

def calcula_matriz_K(A):
    n = A.shape[0]
    #Genero una matriz cuadrada K con las mismas dimensiones que A
    K = np.zeros((n, n))
    for i in range(n):
        #Sumo las conexiones de cada fila de A
        valor_fila = A[i].sum()
        #Agrego en la posición correspondiente de la diagonal de K la suma de los valores
        K[i, i] = valor_fila
    return K

def calcula_matriz_inversa(A):
  #Primero veo si es invertible:
  if np.linalg.det(A) == 0:
    return ('Matriz no invertible')
  else:
    n = A.shape[0]
    I = np.eye(n) #matriz identidad
    AI = np.zeros((n, 2 * n)) #matriz ampliada
    for i in range(n):
        for j in range(n):
            AI[i][j] = A[i][j]
            AI[i][j + n] = I[i][j]
    #Triangulación de Gauss
    for i in range(n):
          pivote = AI[i][i]
          AI[i] = AI[i] / pivote  #Normalizamos la fila i

          for j in range(n):
              if j != i:
                  AI[j] = AI[j] - AI[j][i] * AI[i]
    A_inv = AI[:, n:]

    return A_inv

def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    K = calcula_matriz_K(A)
    Kinv = calcula_matriz_inversa(K)
    #Calculo A transpuesta
    AT = A.T
    C = AT @ Kinv
    # Retorna la matriz C
    return C
    
def calcula_matriz_Kinv(matriz):
    #suma por fila de A
    A = 
    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = ... # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = ...
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = ... # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    n = C.shape[0]
    B = np.eye(n)
    C_potencia = np.eye(n)  # C^0 = I
    for _ in range(1, cantidad_de_visitas):  # desde k = 1 hasta r - 1
        C_potencia = C_potencia @ C  # actualiza C^k
        B += C_potencia
    return B
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
   
