from src.models.core.system import System
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.funcs.olapcube import OLAPCube
from typing import List, Tuple
import numpy as np
from itertools import product

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        
    def convertir_ncubos_a_olapcubes(self, subsistema: System) -> List[OLAPCube]:
        """
        Convierte cada NCube en el objeto System a una instancia OLAPCube.
        
        Returns:
            List[OLAPCube]: Lista de cubos OLAP llenos con los datos de cada NCube.
        """
        olap_cubos = []

        for ncubo in subsistema.ncubos:
            num_dims = len(ncubo.dims)
            olap = OLAPCube(num_dims)

            # Llenar con los datos del NCube
            olap.data = ncubo.data.copy()  # Copia segura del arreglo numpy
            olap_cubos.append(olap)

        return olap_cubos
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        olap_cubos = self.convertir_ncubos_a_olapcubes(self.sia_subsistema)
        
        for i, cubo in enumerate(olap_cubos):
            print(f"\nCalculando tabla de costos para OLAPCube {i}")
            T = self.calcular_tabla_costos(cubo)
            for (origen, destino), costo in T.items():
                print(f"T[{origen} → {destino}] = {round(costo, 3)}")


    def obtener_vecinos(self, estado: Tuple[int]) -> List[Tuple[int]]:
        """Devuelve los estados adyacentes en el hipercubo (distancia de Hamming = 1)."""
        vecinos = []
        for i in range(len(estado)):
            vecino = list(estado)
            vecino[i] = 1 - vecino[i]
            vecinos.append(tuple(vecino))
        return vecinos

    def calcular_tabla_costos(self, cubo: OLAPCube) -> dict:
        """
        Implementa el Algoritmo 1 para calcular la tabla de costos T[i, j] = t(i, j)
        usando BFS modificado.
        
        Returns:
            dict: Diccionario con claves (i, j) y valores t(i, j)
        """
        T = {}
        n = cubo.num_dims
        todos_los_estados = list(product([0, 1], repeat=n))  # Todos los vértices del hipercubo

        for i in todos_los_estados:
            for j in todos_los_estados:
                if i == j:
                    T[(i, j)] = 0.0
                    continue

                d = OLAPCube.hamming(i, j)
                gamma = 2 ** (-d)
                t_ij = abs(cubo.data[i] - cubo.data[j])  # Contribución directa
                T[(i, j)] = t_ij

                if d > 1:
                    Q = [i]
                    visited = {i}
                    level = 0

                    while level < d and Q:
                        level += 1
                        next_Q = []

                        for u in Q:
                            for v in self.obtener_vecinos(u):
                                if v not in visited and OLAPCube.hamming(v, j) < OLAPCube.hamming(u, j):
                                    T[(i, j)] += gamma * T.get((i, v), 0)
                                    visited.add(v)
                                    next_Q.append(v)

                        Q = next_Q

        return T







        
        
#     def find_mip(self):
#     """
#     Implementa el algoritmo para encontrar la bipartición óptima
#     utilizando el enfoque geométrico-topológico.
#     """
#     # 1. Construir la representación n-dimensional del sistema
#     # 2. Calcular la tabla de costos T para cada variable
#     # 3. Identificar las biparticiones candidatas
#     # 4. Evaluar y seleccionar la bipartición óptima
#     # 5. Retornar el resultado en formato compatible