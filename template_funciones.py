# TP-Museos-ALC
# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
#%%
# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')

# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa),
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()

#%%
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


#%%
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

#%%
def calcula_matriz_inversa_1(A):
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

def calcula_matriz_inversa(A):
    # vemos si es invertible
    if np.linalg.det(A) == 0:
        return ('Matriz no invertible')

    n = A.shape[0]
    I = np.eye(n)  # matriz identidad
    AI = np.hstack((A, I))  #matriz ampliada A|I

    for i in range(n):
        #busco el valor máximo absoluto en la columna i en caso de tener que permutar filas
        fila_max = np.argmax(np.abs(AI[i:n, i])) + i  # Encuentro la fila con el pivote máximo
        if fila_max != i:
            # Permutamos las filas i y fila_max en AI
            AI[[i, fila_max], :] = AI[[fila_max, i], :]
        pivote = AI[i, i]

        AI[i] = AI[i] / pivote #hacemos que el pivote sea 1
        # Eliminación
        for j in range(n):
            if j != i:
                AI[j] = AI[j] - AI[j, i] * AI[i]

        A_inv = AI[:, n:] #nos quedamos con la parte derecha de la matriz A|I

    return A_inv
#%%
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
 #%%   
def calculaLU(A):
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    if m!=n:
        print('Matriz no cuadrada')
        return
    for j in range (n):
        for i in range(j+1,n):
            Ac[i,j] = Ac[i,j] / Ac[j,j]
            for k in range(j + 1, n):
                Ac[i, k] = Ac[i, k] - Ac[i, j] * Ac[j, k]
    L = np.tril(Ac,-1) + np.eye(m)
    U = np.triu(Ac)

    return L, U
#%%
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # alfa: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    # Obtenemos el número de museos N a partir de la estructura de la matriz A
    N = A.shape[0]
    I = np.eye(N)
    M = (N/alfa) * (I - (1 - alfa) * C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    # Vector de 1s, multiplicado por el coeficiente correspondiente usando alfa y N.
    b = np.ones(N)*(alfa /N)
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

#%%
def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de transiciones C
    # Retorna la matriz C en versión continua
    D = D.copy()
    #Aplica la función F: f(dji​)=dji​**-1​
    F = 1 / D
    #Asegura que la diagonal sea 0
    np.fill_diagonal(F, 0)
    # Calcula la matriz K, que tiene en su diagonal la suma por filas de F 
    K = calcula_matriz_K(F)
    #Calcula la inversa de K
    Kinv = calcula_matriz_inversa(K)    
    C = Kinv @ F                        # Multiplicación de matrices
    return C

#%%
def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz C de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    n = C.shape[0]
    B = np.eye(n)
    C_potencia = np.eye(n)  # C^0 = I
    for _ in range(1, cantidad_de_visitas):  # desde k = 1 hasta r - 1
        C_potencia = C_potencia @ C  # actualiza C^k
        B += C_potencia
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    return B

#%%
m = 3
A = construye_adyacencia(D, m)
K = calcula_matriz_K(A)
Kinv = calcula_matriz_inversa(K)
Kinv = calcula_matriz_inversa(K)
C = calcula_matriz_C(A)

#%%EJERCICIO 3

m = 3
alfa = 1/5
A = construye_adyacencia(D, m)
p = calcula_pagerank(A, alfa)


