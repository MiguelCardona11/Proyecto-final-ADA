import time
import os
import concurrent.futures
from src.models.core.ncube import NCube
from src.models.core.system import System
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from typing import List, Tuple, Callable
import numpy as np
from itertools import product, combinations
import pandas as pd
from src.funcs.format import fmt_biparticion
from src.funcs.base import seleccionar_metrica
from src.models.base.application import aplicacion
from src.models.core.solution import Solution
from src.constants.base import (
    EFECTO,
    ACTUAL,
)
from src.constants.models import (
    DUMMY_ARR,
)

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self._memoria_costos = {}
        self.distancia_metrica: Callable = seleccionar_metrica(aplicacion.distancia_metrica)
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        tabla = self.calcular_tabla_costos(self.sia_subsistema)
        # print(tabla)
        # self.mostrar_tabla_costos(tabla, tuple(self.sia_subsistema.estado_inicial))
        
        
        # **** PRUEBA CALCULAR COSTO DE TRANSICIÓN ****
        # origen = (0, 0, 0)
        # hammings = self.generar_estados_hamming(origen, 3)
        # print(hammings)
        # destino = (0, 1, 1)
        # mecanismo_str = self.sia_mecanismo_str
        # mascara_presentes = np.array([bit == "1" for bit in mecanismo_str], dtype=bool)

        # for i, ncubo in enumerate(self.sia_subsistema.ncubos):
        #     print(f"\nCostos desde {origen} para NCube {i} (índice {ncubo.indice}):")
        #     costo = self.calcular_costo_transicion(ncubo, origen, destino, mascara_presentes)
        #     print(f"T[{origen} → {destino}] = {costo}")
        
        #return
        
        return self.identificar_biparticiones_candidatas(tabla)
    
    def calcular_tabla_costos(self, subsistema: System) -> np.ndarray:
        n_ncubos = len(subsistema.ncubos)
        
        mecanismo_str = self.sia_mecanismo_str
        mascara_presentes = np.array([bit == "1" for bit in mecanismo_str], dtype=bool) # dice que posiciones de "mecanismo" son equivalen a 1, osea True
        cantidad_presentes = np.count_nonzero(mascara_presentes) # cuenta cuantas posiciones son True
        
        total_estados = 2 ** cantidad_presentes
        estado_inicial = tuple(self.sia_subsistema.estado_inicial[mascara_presentes])
        # print(f"estado inicial --> {estado_inicial}")

        #                       filas     columnas
        tabla_costos = np.full((n_ncubos, total_estados), fill_value=np.nan, dtype=np.float32)
        
        posicion_inicial = self.binario_a_entero(estado_inicial)
        tabla_costos[:, posicion_inicial] = 0.0

        for i, ncubo in enumerate(subsistema.ncubos):
            for j in range(1, len(estado_inicial) + 1):
                hammings = self.generar_estados_hamming(estado_inicial, j)
                if j > 1:
                    for estado in hammings:
                        posicion = self.binario_a_entero(estado)
                        gamma = 2.0 ** (-j)
                        t_ij = abs(ncubo.data[estado_inicial] - ncubo.data[estado])
                        
                        vecinos = self.vecinos_optimos_destino(estado_inicial, estado)
                        sumatoria = 0.0
                        for vecino in vecinos:
                            pos_vecino = self.binario_a_entero(vecino)
                            sumatoria += tabla_costos[i, pos_vecino]
      
                        costo = gamma * (t_ij + sumatoria)
                        
                        tabla_costos[i, posicion] = costo
                else:     
                    for estado in hammings:
                        posicion = self.binario_a_entero(estado)
                        gamma = 2.0 ** (-j)
                        t_ij = abs(ncubo.data[estado_inicial] - ncubo.data[estado])
                        costo = gamma * t_ij
                        
                        tabla_costos[i, posicion] = costo
            
        return tabla_costos
    
    def vecinos_optimos_origen(self, origen: Tuple[int, ...], destino: Tuple[int, ...]) -> List[Tuple[int, ...]]:
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
    
    def vecinos_optimos_destino(self, origen: Tuple[int, ...], destino: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Devuelve los vértices inmediatamente vecinos del vértice destino que se encuentran en algún
        camino óptimo desde el vértice origen hacia el vértice destino.
        """
        n = len(destino)
        vecinos = []

        distancia_actual = self.hamming_distance(origen, destino)

        for i in range(n):
            vecino = list(destino)
            vecino[i] = 1 - vecino[i]  # flip bit i
            vecino_tupla = tuple(vecino)

            nueva_distancia = self.hamming_distance(origen, vecino_tupla)

            if nueva_distancia < distancia_actual:
                vecinos.append(vecino_tupla)

        return vecinos

    def identificar_biparticiones_candidatas(self, tabla: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:

        n_estados = tabla.shape[1] # cantidad de columnas, en este caso son los estados del presente
        n_bits = int(np.log2(n_estados))
        
        # en estado_inicial_bin solo se tienen en cuenta los bits en 1 del mecanismo del subsistema
        mascara_mecanismo = np.array([bit == "1" for bit in self.sia_mecanismo_str], dtype=bool)
        estado_filtrado = self.sia_subsistema.estado_inicial[mascara_mecanismo]
        estado_inicial_bin = ''.join(str(b) for b in estado_filtrado)
        # print(f"Estado inicial real: {estado_inicial_bin}")
        estado_inicial_int = self.binario_str_a_entero(estado_inicial_bin)
        # print(f"Binario: {estado_inicial_bin} igual a {estado_inicial_int} entero ")
        
        mecanismo_str = self.sia_mecanismo_str
        alcance_str = self.sia_alcance_str
        # indices en donde mecanismo y alcance tienen bits iguales a 1 (variables a considerar del subsistema)
        indices_mecanismo = [i for i, bit in enumerate(mecanismo_str) if bit == "1"]
        indices_alcance = [i for i, bit in enumerate(alcance_str) if bit == "1"]
        
        # inicializacion de variables de solucion
        emd_value = 1.0
        mejor_emd = float('inf')
        mejor_dist_marg = DUMMY_ARR
        biparticion_formateada = None
        
        # identificar los ganadores como strings binarios en little-endian
        estado = 0
        while (emd_value != 0.0 and (estado <= int(n_estados/2))):
            complemento = estado ^ (n_estados - 1)
            fila_1 = tabla[:, estado]
            fila_2 = tabla[:, complemento]
            
            # Todas combinaciones binarias posibles (0 = estado, 1 = complemento)
            candidatos = product([0, 1], repeat=len(fila_1))

            mejor_total = float('inf')
            mejores_ganadores = []

            for config in candidatos:
                ganador_actual = []
                total = 0.0
                for i, elegir_complemento in enumerate(config):
                    if elegir_complemento:
                        ganador_actual.append(complemento)
                        total += fila_2[i]
                    else:
                        ganador_actual.append(estado)
                        total += fila_1[i]

                es_igual_al_inicio = all(idx == estado_inicial_int for idx in ganador_actual)

                # Solo considerar candidatos válidos
                if not es_igual_al_inicio:
                    if total < mejor_total:
                        mejor_total = total
                        mejores_ganadores = [list(ganador_actual)]
                    elif total == mejor_total:
                        mejores_ganadores.append(list(ganador_actual))
                        
            # evitar la biparticion donde todos los indices son iguales al estado inicial o sean ganadores triviales
            for indices_ganadores in mejores_ganadores:
                ganador = tuple(format(i, f'0{n_bits}b')[::-1] for i in indices_ganadores)
                
                # construir candidato a a partir de los strings binarios
                referencia = ganador[0]
                arr_alcance_prim = []
                arr_mecanismo_prim = []
                
                arr_alcance_dual = []
                arr_mecanismo_dual = []
                
                # construccion de la biparticion prim
                for idx, actual in enumerate(ganador):
                    if actual == referencia: # [000 111 111]
                        arr_alcance_prim.append(indices_alcance[idx])
                        for i in range(n_bits):
                            if estado_inicial_bin[i] == actual[i]:
                                idx_real = indices_mecanismo[i]
                                if idx_real not in arr_mecanismo_prim:
                                    arr_mecanismo_prim.append(idx_real)
                
                emd_value, dist_marg = self.calcular_emd(arr_alcance_prim, arr_mecanismo_prim)
                # print(f"Ganador: {ganador} con EMD = {emd_value}")
                
                if emd_value < mejor_emd:
                    # print(f"eligió a: {ganador}")
                    mejor_emd = emd_value
                    mejor_dist_marg = dist_marg
                    
                    # construir segunda biparticion a partir del complemento de la primera (solo si el emd_value de la particion prim es óptimo)
                    todas_alcance = set(indices_alcance)
                    alcance_asignado = set(arr_alcance_prim)
                    no_asignadas_alcance = todas_alcance - alcance_asignado
            
                    todas_mecanismo = set(indices_mecanismo)
                    mecanismo_asignado = set(arr_mecanismo_prim)
                    no_asignadas_mecanismo = todas_mecanismo - mecanismo_asignado

                    arr_mecanismo_dual.extend(no_asignadas_mecanismo)
                    arr_alcance_dual.extend(no_asignadas_alcance)
                    
                    # if arr_alcance_dual and arr_mecanismo_dual:
                    # print(f"Evaluando bipartición: {arr_alcance_prim}, {arr_mecanismo_prim} | {arr_alcance_dual}, {arr_mecanismo_dual} -> EMD: {emd_value:.4f}")
                
                    # formatear biparticiones para construccion de la biparticion solucion
                    subalcance_prim = tuple(arr_alcance_prim)
                    submecanismo_prim = tuple(arr_mecanismo_prim)
                    subalcance_dual = tuple(arr_alcance_dual)
                    submecanismo_dual = tuple(arr_mecanismo_dual)
                    
                    biparticion_prim = submecanismo_prim, subalcance_prim
                    biparticion_dual = submecanismo_dual, subalcance_dual
                    
                    biparticion_formateada = fmt_biparticion(
                        [biparticion_prim[ACTUAL], biparticion_prim[EFECTO]],
                        [biparticion_dual[ACTUAL], biparticion_dual[EFECTO]],
                    )
                    
            estado += 1
        
        return Solution(
            estrategia="Geometric",
            perdida=mejor_emd,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist_marg,
            particion=biparticion_formateada,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            hablar=False
        )

    def calcular_emd(self, arr_alcance_prim: np.ndarray, arr_mecanismo_prim: np.ndarray):
        subsistema = self.sia_subsistema
        distribucion_original = self.sia_dists_marginales
        
        particion = subsistema.bipartir(arr_alcance_prim, arr_mecanismo_prim)
        part_marg_dist = particion.distribucion_marginal()
        emd_value = self.distancia_metrica(part_marg_dist, distribucion_original)
        
        return emd_value, part_marg_dist

    def hamming_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        return sum(x != y for x, y in zip(a, b))
    
    def binario_a_entero(self, bits: Tuple[int, ...]) -> int:
        """Convierte una tupla binaria en entero (asumiendo little endian)."""
        return sum(b << i for i, b in enumerate(reversed(bits)))
    
    def binario_str_a_entero(self, bits_str: str) -> int:
        """Convierte un string binario (ej: '0101') a entero en orden little-endian (bit menos significativo primero)."""
        return sum(int(b) << i for i, b in enumerate(reversed(bits_str)))

    def generar_estados_hamming(self, estado: Tuple[int, ...], distancia: int) -> List[Tuple[int, ...]]:
        """
        Genera todos los estados que tienen una distancia de Hamming exacta respecto a `estado`.

        Args:
            estado (Tuple[int, ...]): Estado base como tupla de bits (0 o 1).
            distancia (int): Número de bits que deben diferir con respecto al estado base.

        Returns:
            List[Tuple[int, ...]]: Lista de estados con distancia de Hamming igual a `distancia`.

        Raises:
            ValueError: Si la distancia es inválida (negativa o mayor al número de bits del estado).
        """
        n = len(estado)

        if not (0 <= distancia <= n):
            raise ValueError(f"La distancia de Hamming debe estar entre 0 y {n}, pero se recibió {distancia}.")

        estados_generados = []

        # Generar todas las combinaciones de índices donde se hará flip de bits
        for indices in combinations(range(n), distancia):
            nuevo_estado = list(estado)
            for i in indices:
                nuevo_estado[i] = 1 - nuevo_estado[i]  # flip bit
                
            estados_generados.append(tuple(nuevo_estado))
        return estados_generados
 
    def mostrar_tabla_costos(self, tabla: np.ndarray, estado_inicial: Tuple[int, ...]):
        mecanismo = self.sia_mecanismo_str
        n = mecanismo.count("1")
        
        estados_bin = [format(i, f'0{n}b')[::-1] for i in range(2**n)]  # Reverso para Little Endian [::-1]
        variables = [f'Variable {i}' for i in range(tabla.shape[0])]

        df = pd.DataFrame(tabla.T, index=estados_bin, columns=variables)
        print(df)





