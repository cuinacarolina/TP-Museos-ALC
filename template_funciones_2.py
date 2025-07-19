#PresentaciÃ³n del grupo

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materia: Álgebra Lineal Computacional 
Trabajo Practico 2
Equipo: Brasil
Autores: Carolina Julia Cuina, Juana Gala Moran, MarÃ­a Juliana Salfity
"""
#%%
import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geogrÃ¡ficas
import networkx as nx # ConstrucciÃ³n de la red en NetworkX
from scipy.linalg import solve_triangular
import template_funciones as tp1
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
# Leemos el archivo, retenemos aquellos museos que estÃ¡n en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
# En esta lÃ­nea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interÃ©s, extraemos su geometrÃ­a (los puntos del mapa),
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
# Leemos el archivo vector w
w = pd.read_csv("visitas.txt", sep="\t", header=None).values.flatten()


#%%
#Calcula el Laplaciano no normalizado de una matriz de adyacencia A.
# Recibe:
# A = matriz de adyacencia
# Retorna:
# K - A = matriz Laplaciana donde K es la matriz diagonal de grados

def calcula_L(A):
    K = tp1.calcula_matriz_K(A)  #Calcula la matriz de grados (diagonal)
    return K - A  #Laplaciano: K - A

#%%
# Calcula la matriz P del modelo nulo para modularidad.
# Recibe:
# A = matriz de adyacencia del grafo 
# Retorna:
# P = matriz de probabilidades esperadas de conexión entre nodos

def calcula_P(A):
    n = A.shape[0]
    P = np.zeros((n, n))
    grados = np.sum(A, axis=1)  #Grado de cada nodo
    conexiones = np.sum(A)  #Suma total de todas las conexiones 

    #Construimos la matriz P según el modelo de red aleatoria con mismo grado
    for i in range(n):
        for j in range(n):
            P[i][j] = (grados[i] * grados[j]) / conexiones
    return P
#%%
# Calcula la matriz R = A - P utilizada para el análisis de modularidad.
# Esta matriz permite comparar las conexiones reales con las esperadas por azar.
# Recibe:
# A = matriz de adyacencia del grafo (numpy.ndarray)
# Retorna:
# R = matriz de modularidad R = A - P

def calcula_R(A):
    P = calcula_P(A)  #Calculamos la matriz de conexiones esperadas
    return A - P  #Modularidad: conexiones reales - esperadas

#%%
# Calcula el vector s asociado a un autovector v, asignando 1 o -1 según el signo de cada componente.
# Recibe:
# v = autovector (numpy.ndarray) del cual se extrae la información de signos
# Retorna:
# s = vector columna con valores en {1, -1} según el signo de cada componente de v

def calcula_s(v):
    aux = []
    for coordenada in v:
        if coordenada >= 0:
            aux.append(1)
        else:
            aux.append(-1)
    s = np.array(aux).reshape(-1, 1)  #Convertimos a vector columna
    return s

# Calcula el valor de lambda asociado al vector s y a la matriz Laplaciana L.
# Esta métrica es utilizada en problemas de corte de grafos.
# Recibe:
# L = matriz Laplaciana (numpy.ndarray)
# v = autovector a partir del cual se calcula el vector s
# Retorna:
# lambdaa = valor escalar asociado a sᵗ L s / 4

def calcula_lambda(L, v):
    s = calcula_s(v)  #Vector s con entradas ±1 según el signo de v
    lambdaa = (1/4) * (s.T @ L @ s)  #Fórmula del valor lambda
    return lambdaa

# Calcula el valor de Q (modularidad) para el vector s y la matriz R.
# Es usado para medir la calidad de una partición en comunidades.
# Recibe:
# R = matriz de modularidad R = A - P (numpy.ndarray)
# v = autovector a partir del cual se calcula el vector s
# Retorna:
# Q = valor escalar de modularidad Q = sᵗ R s

def calcula_Q(R, v):
    s = calcula_s(v)  #Vector s con ±1 según el signo de v
    Q = s.T @ R @ s  #Cálculo de la modularidad
    return Q

#%%
# Aplica el método de la potencia para calcular el autovalor de mayor módulo de una matriz A.
# Recibe:
# A = matriz sobre la que se desea aplicar el método
# tol = tolerancia relativa para el criterio de convergencia (por defecto 1e-8)
# maxrep = máximo número de repeticiones permitidas (por defecto infinito)
# Retorna:
# v1 = autovector estimado asociado al autovalor dominante
# l = autovalor dominante estimado
# nrep < maxrep = booleano que indica si el método convergió dentro de maxrep iteraciones

def metpot1(A, tol=1e-8, maxrep=1000, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        
    n, _ = A.shape
    v = np.random.uniform(-1,1,n)  # Generamos un vector inicial aleatorio
    v = v / np.linalg.norm(v)  # Normalización del vector inicial

    v1 = A @ v  # Primera iteración: multiplicación por A
    v1 = v1 / np.linalg.norm(v1)  # Normalización del nuevo vector
    l = v @ (A @ v)  # Estimación inicial del autovalor (producto de Rayleigh)
    
    nrep = 0  # Contador de iteraciones
    l1 = v1 @ (A@v1) # Nueva estimación del autovalor en la siguiente iteración

    # Iteramos hasta que el error relativo sea menor que la tolerancia o hasta alcanzar el máximo de repeticiones
    while np.abs(l1 - l) / np.abs(l) > tol and nrep < maxrep:
        v = v1  # Actualizamos el vector
        l = l1         # Actualizamos el autovalor anterior
        v1 = A @ v1    # Nueva multiplicación
        v1 = v1 / np.linalg.norm(v1)  # Normalización
        l1 = v1 @ (A @ v1)  # Nuevo estimado del autovalor
        nrep += 1  # Incrementamos el contador

    if not nrep < maxrep:
        print('MaxRep alcanzado')  # Por si no convergió

    return v1, l, nrep < maxrep  # Devolvemos el autovector, el autovalor y si se logró la convergencia


#%%
# Aplica el método de deflación para eliminar la contribución del autovalor dominante de una matriz.
# Recibe:
# A = matriz cuadrada sobre la que se desea aplicar la deflación
# tol = tolerancia para la convergencia del método de la potencia (por defecto 1e-8)
# maxrep = máximo número de iteraciones para el método de la potencia (por defecto infinito)
# Retorna:
# deflA = matriz deflacionada, es decir, A sin el componente asociado al autovalor dominante

def deflaciona(A, tol=1e-8, maxrep=np.inf, seed: int = None):
    v1, l1, _ = metpot1(A, tol, maxrep,seed)  #Obtenemos el autovector y autovalor dominante usando el método de la potencia
    deflA = A - l1 * np.outer(v1, v1)  #Aplicamos la fórmula de deflación: A - λ * v * vᵗ
    return deflA

#%%
# Aplica deflación sobre la matriz A para estimar el segundo autovalor y su autovector usando el método de la potencia.
# Recibe:
# A = matriz cuadrada a analizar
# v1 = autovector dominante de A 
# l1 = autovalor dominante de A 
# tol = tolerancia para el criterio de convergencia (por defecto 1e-8)
# maxrep = máximo número de repeticiones permitidas (por defecto infinito)
# Retorna:
# v, l, maxrep = autovector estimado, autovalor asociado y si el método de la potencia convergió

def metpot2(A, v1, l1, tol=1e-8, maxrep=np.inf, seed: int = None):
    deflA = A - l1 * np.outer(v1, v1)  #Deflación: eliminamos la contribución del autovalor dominante
    return metpot1(deflA, tol, maxrep, seed = seed)


#%%
# Aplica el método de la potencia inversa para encontrar el menor autovalor (más chico en módulo) de A + mu*I.
# Recibe:
# A = matriz cuadrada 
# mu = parámetro de desplazamiento 
# tol = tolerancia para el criterio de convergencia (por defecto 1e-8)
# maxrep = máximo número de repeticiones permitidas (por defecto infinito)
# Retorna:
# v, l, maxrep = autovector estimado, autovalor asociado (de A + mu*I), y si el método convergió

def metpotI(A, mu, tol=1e-8, maxrep=np.inf, seed: int = None):
    n, _ = A.shape
    I = np.eye(n)  #Matriz identidad
    X = A + mu * I  #Shift de la matriz
    iX = tp1.calcula_matriz_inversa(X)  #Invertimos A + mu * I
    return metpot1(iX, tol=tol, maxrep=maxrep, seed = seed)  #Aplicamos método de la potencia sobre la inversa


#%%
# Calcula el segundo autovalor más chico (el que sigue al menor, que se asume es 0) y su autovector de la matriz A + mu*I.
# Usamos el método de la potencia inversa combinado con deflación.
# Recibe:
# A = matriz cuadrada 
# mu = parámetro de desplazamiento
# tol = tolerancia para el criterio de convergencia (por defecto 1e-8)
# maxrep = máximo número de repeticiones permitidas (por defecto infinito)
# Retorna:
# v = autovector estimado correspondiente al segundo autovalor más chico
# l = segundo autovalor más chico de A (ajustado después del inverso y del shift)
# exito = booleano que indica si el método de la potencia convergió

def metpotI2(A, mu, tol=1e-8, maxrep=np.inf, seed: int = None):
    n, _ = A.shape
    I = np.eye(n)  
    X = A + mu * I  #sumamos mu (pues conserva los autovalores)
    iX = tp1.calcula_matriz_inversa(X)  #Invertimos A + mu*I
    defliX = deflaciona(iX, tol=tol, maxrep=maxrep, seed = seed)  #Deflación para remover el primer autovalor dominante
    v, l, éxito = metpot1(defliX, tol, maxrep,seed)  #Segundo autovector de la inversa
    l = 1 / l  #Revertimos la inversión
    l -= mu  # "Quitamos" mu para obtener el autovalor original de A
    return v, l, éxito


#%%
# Realiza partición recursiva del grafo usando el método espectral con el Laplaciano.
# Recibe:
# A = matriz de adyacencia 
# niveles = cantidad de divisiones recursivas a realizar
# nombres_s = lista con nombres o índices de los nodos (opcional). Si no se pasa, se usan los índices 0,...,N-1
# Retorna:
# Lista de listas, donde cada sublista contiene los nombres de los nodos de una comunidad.

def laplaciano_iterativo(A, niveles, nombres_s=None,seed: int = None):
    if nombres_s is None:  #Si no se proveyeron nombres, asignamos índices del 0 al N-1
        nombres_s = range(A.shape[0])

    if A.shape[0] == 1 or niveles == 0:
        #Si llegamos a un nodo aislado o a la profundidad deseada, retornamos la comunidad actual
        return [list(nombres_s)]
    else:
        L = calcula_L(A)
        mu = 1e-5
        v, _, _ = metpotI2(L, mu,seed = seed)  # (segundo autovector más chico)

        #Separamos nodos en función del signo del autovector
        ind_p = []  #Índices con coordenadas positivas o cero
        ind_m = []  #Índices con coordenadas negativas
        for i in range(len(v)):
            if v[i] >= 0:
                ind_p.append(i)
            else:
                ind_m.append(i)

        #Si no hay partición posible, retornamos el grupo completo
        if len(ind_p) == 0 or len(ind_m) == 0:
            return [list(nombres_s)]

        #Creamos las submatrices correspondientes a cada grupo
        Ap = A[np.ix_(ind_p, ind_p)]
        Am = A[np.ix_(ind_m, ind_m)]

        #Reorganizamos los nombres de los nodos en cada grupo
        nombres_p = [nombres_s[i] for i in ind_p]
        nombres_m = [nombres_s[i] for i in ind_m]

        #Llamamos recursivamente a la función para seguir particionando
        return laplaciano_iterativo(Ap, niveles - 1, nombres_p,seed = seed) + \
               laplaciano_iterativo(Am, niveles - 1, nombres_m,seed = seed)

#%%
# Realiza partición recursiva del grafo usando el método espectral basado en la matriz de modularidad.
# Recibe:
# A = matriz de adyacencia (opcional)
# R = matriz de modularidad (opcional). Si no se provee, se calcula a partir de A.
# nombres_s = lista con nombres o índices de los nodos (opcional). Si no se pasa, se usan índices 0,...,N-1
# Retorna:
# Lista de listas, donde cada sublista contiene los nombres de los nodos de una comunidad.

def modularidad_iterativo(A=None, R=None, nombres_s=None, seed: int = None):

    #si no se pasa A ni R, no se puede trabajar
    if A is None and R is None:
        print('Dame una matriz')
        return np.nan

    #calculamos la matriz de modularidad si no fue pasada
    if R is None:
        R = calcula_R(A)

    #si no se pasan nombres, usamos índices por defecto
    if nombres_s is None:
        nombres_s = list(range(R.shape[0]))
    else:
        nombres_s = list(nombres_s)

    #si llegamos a un solo nodo, devolvemos la comunidad
    if R.shape[0] == 1:
        return [nombres_s]

    #obtenemos el primer autovector y autovalor de R
    v, l, _ = metpot1(R, seed=seed)

    #calculamos la modularidad actual Q0 de la partición propuesta por el autovector
    Q0 = np.sum(R[v > 0][:, v > 0]) + np.sum(R[v < 0][:, v < 0])

    #si no mejora o no se puede dividir, devolvemos la comunidad entera
    if Q0 <= 0 or all(v > 0) or all(v < 0):
        return [nombres_s]

    #dividimos la matriz R según el signo del autovector
    ind_p = [i for i in range(len(v)) if v[i] > 0]
    ind_m = [i for i in range(len(v)) if v[i] < 0]
    Rp = R[np.ix_(ind_p, ind_p)]
    Rm = R[np.ix_(ind_m, ind_m)]

    #obtenemos los autovectores principales de las submatrices
    vp, lp, _ = metpot1(Rp, seed=seed)
    vm, lm, _ = metpot1(Rm, seed=seed)

    #calculamos la nueva modularidad Q1 tras dividir
    Q1 = 0
    if not all(vp > 0) and not all(vp < 0):
        Q1 += np.sum(Rp[vp > 0][:, vp > 0]) + np.sum(Rp[vp < 0][:, vp < 0])
    if not all(vm > 0) and not all(vm < 0):
        Q1 += np.sum(Rm[vm > 0][:, vm > 0]) + np.sum(Rm[vm < 0][:, vm < 0])

    #si al dividir no se mejora, nos quedamos con la división actual
    if Q0 >= Q1:
        return [
            [nombres_s[i] for i in ind_p],
            [nombres_s[i] for i in ind_m]
        ]

    #si sí mejora, aplicamos recursivamente en cada grupo
    nombres_p = [nombres_s[i] for i in ind_p]
    nombres_m = [nombres_s[i] for i in ind_m]

    return modularidad_iterativo(R=Rp, nombres_s=nombres_p, seed=seed) + \
           modularidad_iterativo(R=Rm, nombres_s=nombres_m, seed=seed)

#%%
# Simetriza una matriz y aplica celilng 
# Recibe:
# A = matriz cuadrada, posiblemente no simétrica y con entradas reales
# Retorna:
# A_moño = matriz simétrica con entradas enteras, donde cada elemento es el techo del promedio simétrico
def simetriza_y_ceiling(A):
    A_sym = 0.5 * (A + A.T) #Promediamos A con su transpuesta para simetrizar
    A_moño = np.ceil(A_sym) #Aplicamos la función np.ceil 
    return A_moño

# Calcula y grafica las comunidades detectadas para distintos valores de m.
# Recibe:
# D: matriz de distancias entre nodos
# lista_m: lista con valores de m para construir A
# metodo: 0 para Laplaciano, 1 para Modularidad
# niveles: cantidad de cortes para el método Laplaciano
# Retorno:
# Grafico: Subplots uno para cada valor de m. Diferenciando con colores las distintas comunidades  
def comunidades_subplot(D, lista_m, metodo):   

    if metodo == 0:
        nombre_metodo = "Laplaciano" 
    else:
        nombre_metodo = "Modularidad"

    fig, axes = plt.subplots(1, len(lista_m), figsize=(9 * len(lista_m), 9))
    if len(lista_m) == 1:
        axes = [axes]

    #Reproyectamos ambos GeoDataFrames
    museos_local = museos.to_crs("EPSG:22184").sort_index()
    barrios_local = barrios.to_crs("EPSG:22184")

    #Calculamos coordenadas de nodos a partir de la geometría reproyectada
    museos_local['x'] = museos_local.geometry.x
    museos_local['y'] = museos_local.geometry.y
    posiciones = {idx: (row['x'], row['y']) for idx, row in museos_local.iterrows()}

    for i, m in enumerate(lista_m):
        #construimos de matriz de adyacencia y comunidad
        A = tp1.construye_adyacencia(D, m)
        A_moño = simetriza_y_ceiling(A)
        niveles = int(np.round(np.log2(len(modularidad_iterativo(A_moño)))))

        if metodo == 0:
            comunidades_detectadas = laplaciano_iterativo(A_moño, niveles, seed=np.random.randint(0, 1000))
        else:
            comunidades_detectadas = modularidad_iterativo(A_moño, seed=np.random.randint(0, 1000))

        ax = axes[i]

        #Dibujamos el mapa de barrios (fondo)
        barrios_local.boundary.plot(ax=ax, color='dimgray', linewidth=0.9)

        #Creamos el grafo
        G = nx.from_numpy_array(A)
        mapping = dict(zip(range(len(museos_local)), museos_local.index))
        G = nx.relabel_nodes(G, mapping)

        colores = plt.get_cmap("tab20", len(comunidades_detectadas))

        #Dibujamos primero las aristas
        nx.draw_networkx_edges(G, posiciones, ax=ax, alpha=0.3, width=1)

        #Dibujamos los nodos por comunidad con borde negro
        for j, grupo in enumerate(comunidades_detectadas):
            nx.draw_networkx_nodes(
                G, posiciones,
                nodelist=grupo,
                node_color=[colores(j)],
                node_size=75,
                ax=ax,
                edgecolors='k',
                linewidths=0.2
            )

        ax.set_title(f"{nombre_metodo} m={m}, comunidades detectadas: {len(comunidades_detectadas)}", fontsize=11)

        # Usar límites del mapa de barrios para que no se corte el gráfico
        bounds = barrios_local.total_bounds  # xmin, ymin, xmax, ymax
        x_min, y_min, x_max, y_max = bounds

        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect("equal")

        ax.axis('off')
    plt.suptitle(f"Comunidades detectadas con método de {nombre_metodo}", fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2)
    plt.show()
