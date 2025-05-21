from src.models.core.ncube import NCube
from src.models.core.system import System
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from typing import List, Tuple, Union
import numpy as np
from itertools import product
import pandas as pd
from collections import defaultdict

DECIMALES_COSTO = 4

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self._memoria_costos = {} 
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        tabla = self.calcular_tabla_costos(self.sia_subsistema)
        self.mostrar_tabla_costos(tabla, tuple(self.sia_subsistema.estado_inicial))
        
        # *** TEST VER BIPARTICIONES CANDIDATAS ***
        biparticiones = self.identificar_biparticiones_candidatas(tabla)
        for i, bip in enumerate(biparticiones):
            print(f"Bipartición {i+1}: {bip}")
        
        # *** TEST VER PARTICIONES FORMADAS ***
        # candidatos = self.identificar_biparticiones_candidatas(tabla)
        # for i, (alcance, mecanismo) in enumerate(candidatos, 1):
        #     print(f"Candidato {i}:")
        #     print("  arr_alcance :", alcance)
        #     print("  arr_mecanismo:", mecanismo)
        #     print()
        
        # **** PRUEBA CALCULAR COSTO DE TRANSICIÓN ****
        # origen = (0, 0, 0)
        # estados = list(product([0, 1], repeat=len(origen)))

        # for i, ncubo in enumerate(self.sia_subsistema.ncubos):
        #     print(f"\nCostos desde {origen} para NCube {i} (índice {ncubo.indice}):")
        #     for destino in estados:
        #         if origen != destino:
        #             costo = self.calcular_costo_transicion(ncubo, origen, destino)
        #             print(f"T[{origen} → {destino}] = {round(costo, 4)}")
                    
        # **** PRUEBA SABER VALORES DE PROBABILIDAD CONDICIONAL ****
        # for i, ncubo in enumerate(self.sia_subsistema.ncubos):
        #     print(f"\nProbabilidades condicionales para NCube {i} (índice {ncubo.indice}):")
        #     for estado in estados:
        #         probabilidad = ncubo.data[estado]
        #         print(f"P[{estado}] = {probabilidad}")
        

    
    def hamming_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        return sum(x != y for x, y in zip(a, b))
    
    def vecinos_optimos(self, origen: Tuple[int, ...], destino: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Devuelve los vértices inmediatamente vecinos del vértice origen que se encuentran en algún camino óptimo hacia el vértice destino."""
        n = len(origen)
        vecinos = []

        distancia_actual = self.hamming_distance(origen, destino)

        for i in range(n):
            vecino = list(origen)
            vecino[i] = 1 - vecino[i]  # flip bit i
            vecino_tupla = tuple(vecino)

            nueva_distancia = self.hamming_distance(vecino_tupla, destino)

            if nueva_distancia < distancia_actual:
                vecinos.append(vecino_tupla)

        return vecinos
    
    def calcular_costo_transicion(self, ncubo: NCube, origen: Tuple[int, ...], destino: Tuple[int, ...]) -> float:
        """Calcula el costo de transición entre un estado origen y un estado destino en un NCubo dado."""
        # Asegurar tipo compacto
        origen = tuple(np.asarray(origen, dtype=np.uint8))
        destino = tuple(np.asarray(destino, dtype=np.uint8))

        # Se revisa si este costo ya se ha calculado
        resultado_costo = (ncubo.indice, origen, destino)
        if resultado_costo in self._memoria_costos:
            return self._memoria_costos[resultado_costo]

        distancia = self.hamming_distance(origen, destino)
        gamma = 2.0 ** (-distancia)
        
        
        t_ij = abs(ncubo.data[origen] - ncubo.data[destino])

        if distancia > 1:
            vecinos_optimos = self.vecinos_optimos(origen, destino)
            costo_vecinos = 0.0
            for vecino in vecinos_optimos:
                costo_vecinos += self.calcular_costo_transicion(ncubo, vecino, destino)
            costo = gamma * (t_ij + costo_vecinos)
        else:
            costo = gamma * t_ij

        # Se redondea resultado y se guarda en memoria
        costo = round(costo, DECIMALES_COSTO)
        self._memoria_costos[resultado_costo] = costo
        return costo
    
    def binario_a_entero(self, bits: Tuple[int, ...]) -> int:
        """Convierte una tupla binaria en entero (asumiendo little endian)."""
        return sum(b << i for i, b in enumerate(reversed(bits)))

    def calcular_tabla_costos(self, subsistema: System) -> np.ndarray:
        """
        Calcula la tabla de costos como matriz NumPy optimizada.

        Returns:
            np.ndarray: Matriz de shape (n_ncubos, 2^n) con los costos desde el estado inicial.
        """
        estado_inicial = tuple(subsistema.estado_inicial)
        n = len(estado_inicial)
        n_ncubos = len(subsistema.ncubos)
        total_estados = 2 ** n

        # Inicializar matriz vacía
        tabla_costos = np.full((n_ncubos, total_estados), np.nan, dtype=np.float32)

        # Precomputar todos los estados posibles en notacion little endian (invertir bits)
        todos_estados = [tuple(reversed(bits)) for bits in product([0, 1], repeat=n)]

        for i, ncubo in enumerate(subsistema.ncubos):
            for destino in todos_estados[1:-1]:  # Excluye el primero y el último
                idx = self.binario_a_entero(destino)
                costo = self.calcular_costo_transicion(ncubo, estado_inicial, destino)
                tabla_costos[i, idx] = round(costo, DECIMALES_COSTO)
        return tabla_costos
    
    def mostrar_tabla_costos(self, tabla: np.ndarray, estado_inicial: Tuple[int, ...]):
        n = len(estado_inicial)
        estados_bin = [format(i, f'0{n}b')[::-1] for i in range(2**n)]  # Reverso para Little Endian
        variables = [f'Variable {i}' for i in range(tabla.shape[0])]

        df = pd.DataFrame(tabla.T, index=estados_bin, columns=variables)
        print(f"Costos desde el estado inicial {estado_inicial}:\n")
        print(df)

    def identificar_biparticiones_candidatas(self, tabla: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_variables = tabla.shape[0]  # cantidad de filas
        n_estados = tabla.shape[1]    # cantidad de columnas
        n_bits = int(np.log2(n_estados))
        ganadores = []

        # Parte 1: obtener ganadores como tuplas de strings binarios invertidos
        for estado in range(n_estados):
            complemento = estado ^ (n_estados - 1)
            if not (estado >= complemento):
                fila_1 = tabla[:, estado]
                fila_2 = tabla[:, complemento]

                if not (np.any(np.isnan(fila_1)) or np.any(np.isnan(fila_2))):
                    indices_ganadores = np.where(fila_1 < fila_2, estado, complemento)
                    ganador_binario = tuple(format(i, f'0{n_bits}b') for i in indices_ganadores)
                    ganadores.append(ganador_binario)

        return ganadores
        # Parte 2: evaluar cada ganador y construir candidatos
        # candidatos: List[Tuple[np.ndarray, np.ndarray]] = []

        # for ganador in ganadores:
        #     valor_referencia = ganador[0]
        #     arr_alcance = [0]
        #     arr_mecanismo = [i for i, bit in enumerate(valor_referencia[::-1]) if bit == '1']

        #     for idx in range(1, len(ganador)):
        #         valor_actual = ganador[idx]
        #         if valor_actual == valor_referencia:
        #             arr_alcance.append(idx)
        #             for i, bit in enumerate(valor_actual[::-1]):
        #                 if bit == '1' and i not in arr_mecanismo:
        #                     arr_mecanismo.append(i)

        #     arr_alcance_np = np.array(arr_alcance, dtype=np.int8)
        #     arr_mecanismo_np = np.array(arr_mecanismo, dtype=np.int8)
        #     candidatos.append((arr_alcance_np, arr_mecanismo_np))

        # return candidatos






        
        
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




