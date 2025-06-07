import time
import os
import concurrent.futures
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
    DUMMY_ARR,
)

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self._memoria_costos = {}
        # self._contador_entraron = 0
        # self._contador_no_entraron = 0
        self.distancia_metrica: Callable = seleccionar_metrica(aplicacion.distancia_metrica)
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        # inicio = time.time()
        tabla = self.calcular_tabla_costos(self.sia_subsistema)
        # fin = time.time()
        # print(f"Tiempo de creacion de tabla: {fin-inicio:.4f}")
        # self.mostrar_tabla_costos(tabla, tuple(self.sia_subsistema.estado_inicial))
        
        # print(f"entraron: {self._contador_entraron}")
        # print(f"NO entraron: {self._contador_no_entraron}")
        
        return self.identificar_biparticiones_candidatas(tabla)
    
    def calcular_tabla_costos(self, subsistema: System) -> np.ndarray:
        """
        Calcula la tabla de costos como matriz NumPy optimizada.

        Returns:
            np.ndarray: Matriz de shape (n_ncubos, 2^n) con los costos desde el estado inicial.
        """
        n_ncubos = len(subsistema.ncubos)
        
        mecanismo_str = self.sia_mecanismo_str
        mascara_presentes = np.array([bit == "1" for bit in mecanismo_str], dtype=bool) # dice que posiciones de "mecanismo" son equivalen a 1, osea True
        cantidad_presentes = np.count_nonzero(mascara_presentes) # cuenta cuantas posiciones son True
        
        total_estados = 2 ** cantidad_presentes
        estado_inicial = tuple(self.sia_subsistema.estado_inicial[mascara_presentes])


        # Inicializar matriz vacía
        tabla_costos = np.zeros((n_ncubos, total_estados), dtype=np.float32) 

        # Precomputar todos los estados posibles en notacion little endian (invertir bits)
        todos_estados = [tuple(reversed(bits)) for bits in product([0, 1], repeat=cantidad_presentes)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for i, ncubo in enumerate(subsistema.ncubos):
                future = executor.submit(self.calcular_costos_ncubo, i, ncubo, todos_estados, estado_inicial, mascara_presentes, tabla_costos)
                futures.append(future)
        
        concurrent.futures.wait(futures)
        return tabla_costos
    
    def calcular_costos_ncubo(self, i, ncubo, todos_estados, estado_inicial, mascara_presentes, tabla_costos):
        for destino in todos_estados:
            if estado_inicial != destino:
                idx = self.binario_a_entero(destino)
                costo = self.calcular_costo_transicion(ncubo, estado_inicial, destino, mascara_presentes)
                tabla_costos[i, idx] = costo 
    
    def calcular_costo_transicion(self, ncubo: NCube, origen: Tuple[int, ...], destino: Tuple[int, ...], mascara: np.ndarray) -> float:
        """Calcula el costo de transición entre un estado origen y un estado destino en un NCubo dado."""

        # self._contador_no_entraron = self._contador_no_entraron + 1
        
        origen = tuple(np.asarray(origen, dtype=np.uint8))
        destino = tuple(np.asarray(destino, dtype=np.uint8))
        
        # Se revisa si este costo ya se ha calculado
        resultado_costo = (ncubo.indice, origen, destino)
        if resultado_costo in self._memoria_costos:
            self._contador_entraron = self._contador_entraron + 1
            return self._memoria_costos[resultado_costo]
        
        distancia = self.hamming_distance(origen, destino)
        gamma = 2.0 ** (-distancia)
        
        # gamma_sumatoria = 2.0 ** (-1)
        
        # se reconstruyen los estados originales utilizando la mascara
        estado_completo_origen = np.zeros_like(mascara, dtype=np.uint8)
        estado_completo_origen[mascara] = origen
        estado_completo_destino = np.zeros_like(mascara, dtype=np.uint8)
        estado_completo_destino[mascara] = destino

        origen_proy = tuple(estado_completo_origen[d] for d in ncubo.dims)
        destino_proy = tuple(estado_completo_destino[d] for d in ncubo.dims)

        t_ij = abs(ncubo.data[origen_proy] - ncubo.data[destino_proy])
        
        # vecinos = self.vecinos(destino)
        # sumatoria = 0.0
        
        # for vecino in vecinos:
        #     vecino = tuple(np.asarray(vecino, dtype=np.uint8))
        #     estado_completo_vecino = np.zeros_like(mascara, dtype=np.uint8)
        #     estado_completo_vecino[mascara] = vecino
        #     vecino_proy = tuple(estado_completo_vecino[d] for d in ncubo.dims)
        #     t_ik = abs(ncubo.data[origen_proy] - ncubo.data[vecino_proy])
        #     sumatoria += gamma_sumatoria*t_ik
        
        # costo = gamma * (t_ij + sumatoria)
        
        # if distancia > 1:
        #     vecinos_j = self.vecinos(destino)
        #     costo_vecinos = 0.0
        #     for k in vecinos_j:
        #         costo_vecinos += self.calcular_costo_transicion(ncubo, origen, k, mascara)
        #     costo = gamma * (t_ij + costo_vecinos)
        
        # else:
        #     costo = gamma * t_ij


        if distancia > 1:
            vecinos_optimos = self.vecinos_optimos(origen, destino)
            costo_vecinos = 0.0
            for vecino in vecinos_optimos:
                costo_vecinos += self.calcular_costo_transicion(ncubo, vecino, destino, mascara)
            costo = gamma * (t_ij + costo_vecinos)
        else:
            costo = gamma * t_ij

        # Se guarda resultado en memoria
        self._memoria_costos[resultado_costo] = costo
        return costo
    
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
    
    def vecinos(self, estado: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Devuelve todos los vecinos inmediatos del estado dado (un solo bit de diferencia)."""
        vecinos = []
        for i in range(len(estado)):
            vecino = list(estado)
            vecino[i] = 1 - vecino[i]  # flip bit i
            vecinos.append(tuple(vecino))
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
                    # print(f"****eligió a: {ganador} = {total_ganador}****")
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
            
    def mostrar_tabla_costos(self, tabla: np.ndarray, estado_inicial: Tuple[int, ...]):
        mecanismo = self.sia_mecanismo_str
        n = mecanismo.count("1")
        
        estados_bin = [format(i, f'0{n}b')[::-1] for i in range(2**n)]  # Reverso para Little Endian [::-1]
        variables = [f'Variable {i}' for i in range(tabla.shape[0])]

        df = pd.DataFrame(tabla.T, index=estados_bin, columns=variables)
        print(f"Costos desde el estado inicial {estado_inicial}:\n")
        print(df)





