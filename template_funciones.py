# TP-Museos-ALC
# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import networkx as nx # Construcción de la red en NetworkX
from scipy.linalg import solve_triangular
#%%
# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa),
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
# Leemos el archivo vector w
w = pd.read_csv("visitas.txt", sep="\t", header=None).values.flatten()

#%%
# La siguiente funcion recibe la matriz de distancias(D) y la cantidad de links por nodo(m). 
# Devuelve la matriz de adyacencia (A) como un numpy.
def construye_adyacencia(D,m):
    #copia D
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexa vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # elege todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convierte a entero
    np.fill_diagonal(A,0) # Borra diagonal para eliminar autolinks
    #retorna A
    return(A)

#%%
#Recibe una matriz A y devuelve una matriz K
def calcula_matriz_K(A):
    # n cantidad de final de A
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
#Recibe una matriz (A) y devuelve su matriz inversa A^-1
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

#%%
#Recibe una matriz A y retorna una matriz C
def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    K = calcula_matriz_K(A) #calcula K de A llamando a la funcion calcula_matriz_k
    Kinv = calcula_matriz_inversa(K) #calcula la inversa de K llamando a la función calcula_matriz_inversa
    #Calculo A transpuesta
    AT = np.transpose(A) 
    C = AT @ Kinv #C producto de A trnaspuesta y la inversa de K
    # Retorna la matriz C
    return C
 #%%   
 #funcion que recibe una matriz A.
 #devuelve la matriz L (parte inferior / lower) y la matriz U (parte superior / upper)
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
    L = np.tril(Ac,-1) + np.eye(m) #toma la parte inferior de Ac sin la diagonal y le suma la identidad para que la diagonal de L sea toso 1
    U = np.triu(Ac) #toma la parte superior de Ac incluyendo la diagonal
    #retorna L, U
    return L, U
#%%
# Función para calcular PageRank usando LU
# A: Matriz de adyacencia
# alfa: coeficientes de damping
# Retorna: Un vector p normalizado con los coeficientes de page rank de cada museo
def calcula_pagerank(A,alfa):
    C = calcula_matriz_C(A) #calcula C de A
    # Obtenemos el número de museos N a partir de la estructura de la matriz A
    N = A.shape[0]
    I = np.eye(N) #matriz identidad de nxn
    M = (N/alfa) * (I - (1 - alfa) * C) #construyo M
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    # Vector de 1s, multiplicado por el coeficiente correspondiente usando alfa y N.
    b = np.ones(N)*(alfa /N)
    Up = solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = solve_triangular(U,Up) # Segunda inversión usando U
    #Normalizamos los valores de pagerank
    p_norm = p/norma_1_vector(p)
    #retorno p_norm
    return p_norm

#%%
# Función para calcular la matriz de transiciones C
# Recibe una matriz D
# Retorna la matriz C en versión continua
def calcula_matriz_C_continua(D): 
   #copio D 
    D = D.copy()
    #evita la división por cero en la diagonal
    np.fill_diagonal(D, np.nan) 
    #Aplica la función F: f(dji​)=dji​**-1​
    F = 1 / D
    #volve a "colocar" los 0 en la diagonal 
    np.fill_diagonal(F, 0)
    # Calcula la matriz K, que tiene en su diagonal la suma por filas de F 
    K = calcula_matriz_K(F)
    #Calcula la inversa de K
    Kinv = calcula_matriz_inversa(K)    
    C = Kinv @ F # Multiplicación de matrices
    return C

#%%
# Recibe la matriz C de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
# suponiendo que cada visitante realizó cantidad_de_visitas pasos
# cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
def calcula_B(C,cantidad_de_visitas):
    #cantidad de filas
    n = C.shape[0]
    #matriz de identidad de nxn
    B = np.eye(n)
    C_potencia = np.eye(n)  # C^0 = I
    for _ in range(1, cantidad_de_visitas):  # desde k = 1 hasta r - 1
        C_potencia = C_potencia @ C  # actualiza C^k
        B += C_potencia
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    return B

#%%EJERCICIO 3
def ejercicio_3_a(D):
    m = 3
    alfa = 1/5
    A = construye_adyacencia(D, m)
    p = calcula_pagerank(A, alfa)
    # Creamos el grafo
    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    factor_escala = 1e4
    fig, ax = plt.subplots(figsize=(10, 10))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    nx.draw_networkx(G, G_layout, node_size=p*factor_escala, node_color=p, cmap=plt.cm.viridis,
                 ax=ax, with_labels=False, edge_color='gray', alpha=0.8)
    escala = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    escala.set_array(p)
    plt.colorbar(escala, ax=ax, label='PageRank')
    plt.title("Red de Museos con Tamaños según PageRank")
    plt.show()
    return 


def ejercicio_3_b(D, alpha, rango_m):     
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))  #1 fila, 4 columnas
    
    for i, m in enumerate(rango_m):
        ax = axes[i]    
        A = construye_adyacencia(D, m)
        G = nx.from_numpy_array(A)
        G_layout = {
            i: v for i, v in enumerate(zip(
                museos.to_crs("EPSG:22184").get_coordinates()['x'],
                museos.to_crs("EPSG:22184").get_coordinates()['y']
            ))
        }
        factor_escala = 1e4
        p = calcula_pagerank(A, alpha)
        barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
        nx.draw_networkx(
            G, G_layout, node_size=p * factor_escala, node_color=p,
            cmap=plt.cm.viridis, ax=ax, with_labels=False,
            edge_color='gray', alpha=0.8
            )
        ax.set_title(f'm = {m}')
        plt.suptitle("Red de Museos con Tamaños según PageRank y distintos m", fontsize=18, y=1.02)
        plt.tight_layout()
    plt.show()
    return                                                                                                                                                                                 
       
def ejercicio_3_c(D, m, rango_alpha):
    A = construye_adyacencia(D, m)
    G = nx.from_numpy_array(A)
    G_layout = {
        i: v for i, v in enumerate(zip(
            museos.to_crs("EPSG:22184").get_coordinates()['x'],
            museos.to_crs("EPSG:22184").get_coordinates()['y']
        ))
    }
    factor_escala = 1e4
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # 2 filas, 4 columnas

    for i in range(8):  # siempre 8 subplots
        fila, col = divmod(i, 4)
        ax = axes[fila][col]
        if i < len(rango_alpha):
            valor = rango_alpha[i]
            p = calcula_pagerank(A, valor)
            barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
            nx.draw_networkx(
                G, G_layout, node_size=p * factor_escala, node_color=p,
                cmap=plt.cm.viridis, ax=ax, with_labels=False,
                edge_color='gray', alpha=0.8
            )
            
            ax.set_title(f'α = {valor:.2f}')

        else:
            ax.axis('off')  # espacio vacío
        ax.axis('off')

    plt.suptitle("Red de Museos con Tamaños según PageRank y distintos α", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

    return
    

#%% Museos con mayor pagerank variando el m 
def grafico_mayores_pg_variando_m(D, alpha, rango_m):    
    maximos_indices = set()
    resultados = {}
    for m in rango_m:
        A = construye_adyacencia(D, m)
        p = calcula_pagerank(A, 1/5)
        top3 = np.argsort(p)[-3:][::-1]  #índices de los 3 mayores
        maximos_indices.update(top3)    
        
    for idx in maximos_indices:
        resultados[idx] = []
    
    #volvemos a recorrer rango_m y guardar los pageranks de esos museos
    for m in rango_m:
        A = construye_adyacencia(D, m)
        p = calcula_pagerank(A, 1/5)
        
        for idx in maximos_indices:
            resultados[idx].append(p[idx])
    
    #Graficamos
    plt.figure(figsize=(12, 8))
    
    for idx, valores in resultados.items():
        nombre = museos.loc[idx, "name"]
        plt.plot(rango_m, valores, label=nombre)
    
    plt.xlabel("m [cantidad de vecinos]")
    plt.ylabel("Valor de PageRank")
    plt.title("PageRank de museos más centrales según m")
    plt.legend(fontsize="small", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return

#%% Museos con mayor pagerank variando el alpha

def grafico_mayores_pg_variando_alpha(D, m, rango_alpha):
    A = construye_adyacencia(D, m)

    # Guardamos todos los vectores PageRank
    pageranks = []
    museos_centrales = set()

    for alpha in rango_alpha:
        p = calcula_pagerank(A, alpha)
        pageranks.append(p)

        # Detectar top 3
        top_3 = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:3]
        museos_centrales.update(top_3)

    # Ahora armamos la trayectoria completa de cada museo central
    trayectoria = {idx: [] for idx in museos_centrales}
    for p in pageranks:
        for idx in museos_centrales:
            trayectoria[idx].append(p[idx])

    # Graficar
    plt.figure(figsize=(12, 12))
    for idx, valores in trayectoria.items():
        nombre = museos.loc[idx, "name"]
        plt.plot(rango_alpha, valores, marker ='o', label = nombre)
    

    plt.xlabel('Alpha [factor de amortiguamiento]')
    plt.ylabel('Valor de PageRank')
    plt.title("PageRank de museos más centrales según alpha")
    plt.legend(fontsize="small", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return

#%%EJERCICIO 5
#recibe un vector y devuelve la norma 1 de ese vector
def norma_1_vector(vector):
    #crea una varibale para que almacene la suma de los modulos de las coordenadas del vector
    res = 0
    for elemento in vector:
        #suma los modulos de todas las coordenadas del vector
        res += abs(elemento)
    #retorna la norma 1 del vector
    return res

# esta funcion resuelve la ecuacion v = B^(-1)*w
#recibe una matriz B y retorna la norma de v
def resolucion_eq_5(B):
    L, U = calculaLU(B) #calcula L, U de B
    y = solve_triangular(L, w, lower=True) #encuentra y tal que L*y = w
    v = norma_1_vector(solve_triangular(U, y, lower=False)) 
    #encuentra v tal que U*v = y
    #normaliza el resultado v usando norma 1
    #retorna v normalizado
    return v


#%%EJERCICIO 6
#Recibe una matriz (A) y devuelve la norma1 de esa matriz
def norma_1_matriz(A):
    #crea res una lista vacia
    res = []
    #recorre todas las columnas de A
    for i in range (0,len(A),1):
        #crea una variable que almacena las suma de los modulos de los numeros de la columna.
        suma = 0
        for fila in A:
            suma += abs(fila[i])
        #cuando recorre todas las filas para una columna guardo la suma en res.
        res.append(suma)
    #una vez que tiene las suma de todas las columnas de la matriz devuelve la maxima de ellas.
    return max(res)

#Recibe una matriz (B) y devuelve la condicion de B
def condicion_1(B):
    #calcula la norma 1 de B llamando a la funcion norma_1_matriz.
    norma_B = norma_1_matriz(B) 
    #calcula la inversa de B llamando a la funcion calcula_matriz_inversa(B).
    Binv = calcula_matriz_inversa(B)
    #calcula la norma 1 de la inversa de B llamando a la funcion norma_1_matriz.
    inv_norma = norma_1_matriz(Binv)
    #la variable condicion es la condicion de B es decir el producto entre la norma_B y la inv_norma
    cond = norma_B * inv_norma
    #retorna cond
    return cond






