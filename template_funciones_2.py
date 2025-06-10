#PresentaciÃ³n del grupo

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materia: Ãlgebra Lineal Computacional 
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
#w = pd.read_csv("visitas.txt", sep="\t", header=None).values.flatten()


#%%
def calcula_L(A):
    K = tp1.calcula_matriz_K(A)
    return K - A


#%%
def calcula_P(A):
    n = A.shape[0]
    P = np.zeros((n, n))
    grados = np.sum(A, axis=1)
    conexiones = np.sum(A)  
    #Construimos la matriz P 
    for i in range(n):
        for j in range(n):
            P[i][j] = (grados[i] * grados[j]) / conexiones
    return P

P = calcula_P(A_ejemplo)
#%%
def calcula_R(A):
    P = calcula_P(A)
    return A - P

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

# Tomamos el autovalor mÃ¡s grande y su correspondiente autovector
indice_max = np.argmax(valores)
autovalor_principal = valores[indice_max]
autovector_principal = vectores[:, indice_max]

#lambda_ejemplo = calcula_lambda(L, autovector_principal)

q_ejemplo = calcula_Q(R, autovector_principal)
#%%


def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   n,_ = A.shape
   v = np.random.rand(n) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v/ np.linalg.norm(v) # Lo normalizamos

   v1 = A @ v  #Aplicamos la matriz una vez
   v1 = (v1)/np.linalg.norm(v1) #normalizamos
   l = (v.T @ A @ v) / (v.T @ v) #Calculamos el autovalor estimado
   nrep = 0 #Contador

   l1 = (v1.T @ A @ v1) / (v1.T @ v1) # Y el estimado en el siguiente paso   

   while np.abs(l1 - l) / np.abs(l) > tol and nrep < maxrep:
        norm_v1 = np.linalg.norm(v1)
        if norm_v1 < 1e-12:
            print("Norma casi cero: v1 =", v1)
            return v1, 0, False
        v = v1.copy()
        l = l1
        v1 = A @ v1
        v1 = v1 / np.linalg.norm(v1)
        l1 = (v1.T @ A @ v1) / (v1.T @ v1)

        nrep += 1

   if not nrep < maxrep:
      print('MaxRep alcanzado')
   return v1,l,nrep<maxrep


print(metpot1(A_ejemplo))
#%%
def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el mÃ©todo de la potencia, y un nÃºmero mÃ¡ximo de repeticiones
    v1,l1, _ = metpot1(A,tol,maxrep) # Buscamos primer autovector con mÃ©todo de la potencia
    deflA = A - l1 * np.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

deflaciona_ejemplo = deflaciona(A_ejemplo)

#%%
def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = A - l1 * np.outer(v1, v1)
   return metpot1(deflA,tol,maxrep)

#v1,l1,_ = autovalor_max_ejemplo
#segundo_autovalor_ejemplo = metpot2(A_ejemplo, v1, l1)
#%%
def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el mÃ©todo convergiÃ³.
    #calculo el    v = V[:,0]
    #autovalor la inversa de A + mu * I
    n,_ = A.shape #tomo la dimension de A
    I = np.eye(n) #creo la matriz identidad de dimension n
    X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
    iX = tp1.calcula_matriz_inversa(X)
    return metpot1(iX,tol=tol,maxrep=maxrep)
#%%
def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegÃ³ a converger.
   n,_ = A.shape #tomo la dimension de A
   I = np.eye(n) #creo la matriz identidad de dimension n
   X = A + mu* I # Calculamos la matriz A shifteada en mu (plantilla)
   iX = tp1.calcula_matriz_inversa(X) # La invertimos
   defliX = deflaciona(iX,tol=1e-8,maxrep=np.inf) # La deflacionamos
   v,l,_ = metpot1(defliX,tol,maxrep)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_
#%%
def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La funciÃ³n debe, recursivamente, ir realizando cortes y reduciendo en 1 el nÃºmero de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al Ãºltimo paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v1,l1,_ = metpot1(A,tol=1e-8,maxrep=np.inf)
        v,_,_ = metpot2(L,v1,l1)# Encontramos el segundo autovector de L
        
        # Separamos los nodos en dos grupos segÃºn el signo del segundo autovector
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
def modularidad_iterativo(A=None, R=None, nombres_s=None):
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = list(range(R.shape[0]))
    else:
        nombres_s = list(nombres_s)

    # Criterio de corte
    if R.shape[0] <= 1:
        return [nombres_s]

    # Autovector principal de R
    v, l, _ = metpot1(R)
    L, V = np.linalg.eig(R)
    v = V[:,0]
    print(f"{np.linalg.norm(R @ v - l * v)}")
    # DivisiÃ³n segÃºn signo del autovector
    ind_p = [i for i in range(len(v)) if v[i] >= 0]
    ind_m = [i for i in range(len(v)) if v[i] < 0]
    print(ind_m)
    # Si no se puede dividir, devolver comunidad entera
    if len(ind_p) == 0 or len(ind_m) == 0:
        print("no dividio")
        return [nombres_s]

    # Modularidad ganada por esta divisiÃ³n
    Q = np.sum(R[np.ix_(ind_p, ind_p)]) + np.sum(R[np.ix_(ind_m, ind_m)])

    if Q <= 0:
        return [nombres_s]

    # DivisiÃ³n recursiva
    nombres_p = [nombres_s[i] for i in ind_p]
    nombres_m = [nombres_s[i] for i in ind_m]
    Rp = R[np.ix_(ind_p, ind_p)]
    Rm = R[np.ix_(ind_m, ind_m)]

    return modularidad_iterativo(R=Rp, nombres_s=nombres_p) + \
           modularidad_iterativo(R=Rm, nombres_s=nombres_m)

#modularidad_iterativo(A_ejemplo)
modularidad_iterativo(tp1.construye_adyacencia(D, 3))
#%% pruebaÃ§

A = tp1.construye_adyacencia(D, 3)
comunidades = modularidad_iterativo(A)
print(len(comunidades))  # Â¿Da mÃ¡s de 1?
print([len(c) for c in comunidades])
print("Suma de modularidad esperada:", np.sum(R))
print("Matriz R simÃ©trica:", np.allclose(R, R.T))
#%%

#primero hacemos la funcion celling:
def simetriza_y_ceiling(A):
    A_sym = 0.5 * (A + A.T)         
    A_moño = np.ceil(A_sym)         
    return A_moño

#Calcula y grafica las comunidades detectadas para distintos valores de m.
#ParÃ¡metros:
#  -D: matriz de distancias entre nodos
#  -lista_m: lista con valores de m para construir A
#  -metodo: 0 para Laplaciano, 1 para Modularidad
#  -niveles: cantidad de cortes para el mÃ©todo Laplaciano
  
def comunidades_subplot(D, lista_m, metodo, niveles=4):
    if metodo == 0:
        nombre_metodo = "Laplaciano" 
    else:
        nombre_metodo = "Modularidad"

    fig, axes = plt.subplots(1, len(lista_m), figsize=(7 * len(lista_m), 7))
    if len(lista_m) == 1:
        axes = [axes]  # por si hay un solo subplot

    # Precalculo coordenadas para museos
    museos['x'] = museos.geometry.x 
    museos['y'] = museos.geometry.y
    posiciones = {idx: (row['x'], row['y']) for idx, row in museos.iterrows()}

    for i, m in enumerate(lista_m):
        A = tp1.construye_adyacencia(D, m)
        A_moño = simetriza_y_ceiling(A)

        if metodo == 0:
            comunidades_detectadas = laplaciano_iterativo(A_moño, niveles=niveles)
            
        else:
            comunidades_detectadas = modularidad_iterativo(A_moño)

        ax = axes[i]
        print(axes.shape)
        barrios.to_crs("EPSG:22184").boundary.plot(ax=ax, color='lightgray', linewidth=0.7)
        G = nx.from_numpy_array(A)
        colores = plt.get_cmap("tab20", len(comunidades_detectadas))
        
        print(f"/comunidades={comunidades_detectadas}")

        for j, grupo in enumerate(comunidades_detectadas):
            nx.draw_networkx_nodes(
                G, posiciones,
                nodelist=grupo,
                node_color=[colores(j)],
                node_size=75,
                ax=ax
            )

        nx.draw_networkx_edges(G, posiciones, ax=ax, alpha=0.3, width=0.3)
        ax.set_title(f"{nombre_metodo} m={m}, comunidades detectadas: {len(comunidades_detectadas)}", fontsize=11)

        # Ajuste de límites geográficos
        xs, ys = zip(*posiciones.values())
        ax.set_xlim(min(xs) - 0.01, max(xs) + 0.01)
        ax.set_ylim(min(ys) - 0.01, max(ys) + 0.01)

    plt.suptitle(f"Comunidades detectadas con método de {nombre_metodo}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # deja espacio para el título
    plt.show()


#comunidades_subplot(D, [3,5,10,50], 0)
comunidades_subplot(D, [3,5,10,50], 1)
