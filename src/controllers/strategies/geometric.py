import time
from src.models.core.ncube import NCube
from src.models.core.system import System
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from typing import List, Tuple
import numpy as np
from itertools import product
import pandas as pd
from typing import Callable
from src.funcs.format import fmt_biparticion
from src.funcs.base import seleccionar_metrica
from src.models.base.application import aplicacion
from src.models.core.solution import Solution
from src.constants.base import (
    EFECTO,
    ACTUAL,
)
from src.constants.models import (
    BRUTEFORCE_LABEL,
    DUMMY_ARR,
    DUMMY_EMD,
    ERROR_PARTITION,
)

DECIMALES_COSTO = 4

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self._memoria_costos = {} 
        self.distancia_metrica: Callable = seleccionar_metrica(aplicacion.distancia_metrica)
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        tabla = self.calcular_tabla_costos(self.sia_subsistema)
        self.mostrar_tabla_costos(tabla, tuple(self.sia_subsistema.estado_inicial))
        
        # canditatos = self.identificar_biparticiones_candidatas(tabla)
        # return self.evaluar_candidatos(canditatos)
        

        
    
        
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
        
        # print(f"Calculando costo de transición entre {origen} y {destino} para NCube {ncubo.indice}...")
        # print(f"origen: {origen}, destino: {destino}")
        
        dims = ncubo.dims  # por ejemplo, [0, 2]
        origen_proy = tuple(origen[d] for d in dims)
        destino_proy = tuple(destino[d] for d in dims)

        t_ij = abs(ncubo.data[origen_proy] - ncubo.data[destino_proy])

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
        n_ncubos = len(subsistema.ncubos)
        
        mecanismo = self.sia_mecanismo_str
        cantidad_presentes = mecanismo.count("1")
        total_estados = 2 ** cantidad_presentes


        # Inicializar matriz vacía
        tabla_costos = np.zeros((n_ncubos, total_estados), dtype=np.float32) 

        # Precomputar todos los estados posibles en notacion little endian (invertir bits)
        todos_estados = [tuple(reversed(bits)) for bits in product([0, 1], repeat=cantidad_presentes)]
        # print(todos_estados)

        for i, ncubo in enumerate(subsistema.ncubos):
            for destino in todos_estados:
                if estado_inicial != destino:
                    idx = self.binario_a_entero(destino)
                    costo = self.calcular_costo_transicion(ncubo, estado_inicial, destino)
                    tabla_costos[i, idx] = round(costo, DECIMALES_COSTO)
        return tabla_costos
    
    def mostrar_tabla_costos(self, tabla: np.ndarray, estado_inicial: Tuple[int, ...]):
        mecanismo = self.sia_mecanismo_str
        n = mecanismo.count("1")
        
        estados_bin = [format(i, f'0{n}b')[::-1] for i in range(2**n)]  # Reverso para Little Endian [::-1]
        variables = [f'Variable {i}' for i in range(tabla.shape[0])]

        df = pd.DataFrame(tabla.T, index=estados_bin, columns=variables)
        print(f"Costos desde el estado inicial {estado_inicial}:\n")
        print(df)

    def identificar_biparticiones_candidatas(self, tabla: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_variables = tabla.shape[0]
        n_estados = tabla.shape[1]
        n_bits = int(np.log2(n_estados))
        estado_inicial_bin = format(self.binario_a_entero(self.sia_subsistema.estado_inicial), f'0{n_bits}b')[::-1]

        ganadores = []

        # Parte 1: identificar los ganadores como strings binarios en little-endian
        for estado in range(n_estados):
            complemento = estado ^ (n_estados - 1)
            if estado < complemento:
                fila_1 = tabla[:, estado]
                fila_2 = tabla[:, complemento]

                if not (np.any(np.isnan(fila_1)) or np.any(np.isnan(fila_2))):
                    indices_ganadores = np.where(fila_1 < fila_2, estado, complemento)
                    if not np.all(indices_ganadores == self.binario_a_entero(self.sia_subsistema.estado_inicial)):
                        ganador_binario = tuple(format(i, f'0{n_bits}b')[::-1] for i in indices_ganadores)
                        ganadores.append(ganador_binario)
                        
        return ganadores

        # Parte 2: construir candidatos con lógica de comparación bit a bit con estado inicial
        # candidatos: List[Tuple[np.ndarray, np.ndarray]] = []

        # for ganador in ganadores:
        #     referencia = ganador[0]
        #     arr_alcance = [0]
        #     arr_mecanismo = [
        #         i for i in range(n_bits) if estado_inicial_bin[i] == referencia[i]
        #     ]

        #     for idx in range(1, len(ganador)):
        #         actual = ganador[idx]
        #         if actual == referencia:
        #             arr_alcance.append(idx)
        #             for i in range(n_bits):
        #                 if estado_inicial_bin[i] == actual[i] and i not in arr_mecanismo:
        #                     arr_mecanismo.append(i)

        #     arr_alcance_np = np.array(arr_alcance, dtype=np.int8)
        #     arr_mecanismo_np = np.array(arr_mecanismo, dtype=np.int8)
        #     candidatos.append((arr_alcance_np, arr_mecanismo_np))

        # return candidatos
    
    def evaluar_candidatos(self, candidatos: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        subsistema = self.sia_subsistema
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        distribucion_original = self.sia_dists_marginales
        
        mejor_emd = float('inf')
        
        for arr_alcance, arr_mecanismo in candidatos:
            particion = subsistema.bipartir(arr_alcance, arr_mecanismo)
            part_marg_dist = particion.distribucion_marginal()
            emd_value = self.distancia_metrica(part_marg_dist, distribucion_original)

            if emd_value < mejor_emd:
                mejor_emd = emd_value
                mejor_dist_marg = part_marg_dist
                
                subalcance = tuple(arr_alcance)
                submecanismo = tuple(arr_mecanismo)
                biparticion_prim = submecanismo, subalcance
                biparticion_dual = (
                    set(presentes.data) - set(submecanismo),
                    set(futuros.data) - set(subalcance),
                )
        
        biparticion_formateada = fmt_biparticion(
            [biparticion_prim[ACTUAL], biparticion_prim[EFECTO]],
            [biparticion_dual[ACTUAL], biparticion_dual[EFECTO]],
        )
        
        return Solution(
            estrategia="Geométric",
            perdida=mejor_emd,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist_marg,
            particion=biparticion_formateada,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            hablar=False
        )







        
        
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
#     ANOTACIONES IMPORTANTES:
#       - CAMBIAR TODAS LAS LIST POR TUPLAS




