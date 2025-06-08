import time
import concurrent.futures
import pandas as pd
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
        self.distancia_metrica: Callable = seleccionar_metrica(aplicacion.distancia_metrica)
        
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str):
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        self._memoria_costos = {}
        tabla = self.calcular_tabla_costos(self.sia_subsistema)
        # self.mostrar_tabla_costos(tabla, mecanismo)
        # self.mostrar_tabla_costos_invertida(tabla, mecanismo)
        
        return self.identificar_biparticion(tabla)

    def calcular_costos_ncubo(self, estado_inicial, ncubo, tabla_costos, i):
        for j in range(1, len(estado_inicial) + 1):
            hammings = self.generar_estados_hamming(estado_inicial, j)
            gamma = 2.0 ** (-j)

            for estado in hammings:
                posicion = self.binario_a_entero(estado)
                t_ij = abs(ncubo.data[estado_inicial] - ncubo.data[estado])
                
                sumatoria = 0.0
                if j > 1:
                    vecinos = self.vecinos_optimos_destino(estado_inicial, estado)
                    for vecino in vecinos:
                        pos_vecino = self.binario_a_entero(vecino)
                        sumatoria += tabla_costos[i, pos_vecino]
                        
                costo = gamma * (t_ij + sumatoria)
                tabla_costos[i, posicion] = costo
    
    """
    General tabla de costo de transiciones desde el estado incial de un subsistema, hacia el resto de estados de cada NCubo.
    """
    def calcular_tabla_costos(self, subsistema: System) -> np.ndarray:
        n_ncubos = len(subsistema.ncubos)
        
        mecanismo_str = self.sia_mecanismo_str
        mascara_presentes = np.array([bit == "1" for bit in mecanismo_str], dtype=bool) # dice que posiciones de "mecanismo" son equivalen a 1, osea True
        cantidad_presentes = np.count_nonzero(mascara_presentes) # cuenta cuantas posiciones son True
        
        total_estados = 2 ** cantidad_presentes
        estado_inicial = tuple(self.sia_subsistema.estado_inicial[mascara_presentes])
        #                       filas     columnas
        tabla_costos = np.full((n_ncubos, total_estados), fill_value=np.nan, dtype=np.float32)
        
        posicion_inicial = self.binario_a_entero(estado_inicial)
        tabla_costos[:, posicion_inicial] = 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, ncubo in enumerate(subsistema.ncubos):
                future = executor.submit(self.calcular_costos_ncubo, estado_inicial, ncubo, tabla_costos, i)
                futures.append(future)
                
        concurrent.futures.wait(futures)
        return tabla_costos
    

    def identificar_biparticion(self, tabla: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:

        n_estados = tabla.shape[1] # cantidad de columnas, en este caso son los estados del presente
        n_bits = int(np.log2(n_estados))
        
        # en estado_inicial_bin solo se tienen en cuenta los bits en 1 del mecanismo del subsistema
        mascara_mecanismo = np.array([bit == "1" for bit in self.sia_mecanismo_str], dtype=bool)
        estado_filtrado = self.sia_subsistema.estado_inicial[mascara_mecanismo]
        estado_inicial_bin = ''.join(str(b) for b in estado_filtrado)
        estado_inicial_int = self.binario_str_a_entero(estado_inicial_bin)
        
        mecanismo_str = self.sia_mecanismo_str
        alcance_str = self.sia_alcance_str
        # indices en donde mecanismo y alcance tienen bits iguales a 1 (variables a considerar del subsistema)
        indices_mecanismo = [i for i, bit in enumerate(mecanismo_str) if bit == "1"]
        indices_alcance = [i for i, bit in enumerate(alcance_str) if bit == "1"]
        
        # inicializacion de variables de solucion
        emd_value = 1.0
        encontrado_emd_cero = False
        mejor_emd = float('inf')
        mejor_dist_marg = DUMMY_ARR
        biparticion_formateada = None
        
        estado = 0
        while not encontrado_emd_cero and (estado <= int(n_estados/2)-1):
            # *** BUSQUEDA DE CANDIDATOS ***
            complemento = estado ^ (n_estados - 1)
            fila_1 = tabla[:, estado]
            fila_2 = tabla[:, complemento]
            
            candidatos_parciales = [[]]

            for i in range(len(fila_1)):
                nuevo_conjunto = []
                costo_1 = fila_1[i]
                costo_2 = fila_2[i]
                
                if costo_1 < costo_2:
                    for parcial in candidatos_parciales:
                        nuevo_conjunto.append(parcial + [estado])
                elif costo_2 < costo_1:
                    for parcial in candidatos_parciales:
                        nuevo_conjunto.append(parcial + [complemento])
                else:  # empate, duplicar caminos
                    for parcial in candidatos_parciales:
                        nuevo_conjunto.append(parcial + [estado])
                        nuevo_conjunto.append(parcial + [complemento])
                
                candidatos_parciales = nuevo_conjunto

            # Se retiran candidatos triviales
            candidatos = []

            for candidato_int in candidatos_parciales:
                if not all(idx == estado_inicial_int for idx in candidato_int):
                    candidatos.append(candidato_int)

            # *** EVALUACION DE LOS CANDIDATOS ***
            for candidato_int in candidatos:
                candidato_int = candidato_int[::-1]
                candidato = tuple(format(i, f'0{n_bits}b') for i in candidato_int)
                            
                # construir candidato a a partir de los strings binarios
                referencia = candidato[0]
                arr_alcance_prim = []
                arr_mecanismo_prim = []
                
                arr_alcance_dual = []
                arr_mecanismo_dual = []
                
                # construccion de la biparticion prim
                for idx, actual in enumerate(candidato):
                    if actual == referencia: # [000 111 111]
                        arr_alcance_prim.append(indices_alcance[idx])
                        for i in range(n_bits):
                            if estado_inicial_bin[i] == actual[i]:
                                idx_real = indices_mecanismo[i]
                                if idx_real not in arr_mecanismo_prim:
                                    arr_mecanismo_prim.append(idx_real)
                
                emd_value, dist_marg = self.calcular_emd(arr_alcance_prim, arr_mecanismo_prim)
                
                if emd_value == 0.0:
                    encontrado_emd_cero = True
                                
                # ***CONSTRUCCION DE BIPARTICION
                if emd_value < mejor_emd:
                    mejor_emd = emd_value
                    mejor_dist_marg = dist_marg
                    
                    # construir segunda biparticion a partir del complemento de la primera
                    todas_alcance = set(indices_alcance)
                    alcance_asignado = set(arr_alcance_prim)
                    no_asignadas_alcance = todas_alcance - alcance_asignado
            
                    todas_mecanismo = set(indices_mecanismo)
                    mecanismo_asignado = set(arr_mecanismo_prim)
                    no_asignadas_mecanismo = todas_mecanismo - mecanismo_asignado

                    arr_mecanismo_dual.extend(no_asignadas_mecanismo)
                    arr_alcance_dual.extend(no_asignadas_alcance)
                
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
        estados_generados = []
        # Generar todas las combinaciones de índices donde se hará flip de bits
        for indices in combinations(range(n), distancia):
            nuevo_estado = list(estado)
            for i in indices:
                nuevo_estado[i] = 1 - nuevo_estado[i]  # flip bit
                
            estados_generados.append(tuple(nuevo_estado))
        return estados_generados

    def mostrar_tabla_costos(self, tabla: np.ndarray, mecanismo_str: str):
        """
        Muestra la tabla de costos con índices binarios para las columnas y
        etiquetas de filas para cada ncubo.
        
        Parámetros:
        - tabla: np.ndarray, matriz de forma (n_ncubos, total_estados)
        - mecanismo_str: str, como "10011", para calcular cuántos bits activos hay
        """
        cantidad_presentes = sum(bit == "1" for bit in mecanismo_str)
        total_estados = 2 ** cantidad_presentes
        
        # Generar etiquetas binarios para las columnas
        etiquetas_columnas = [
            format(i, f"0{cantidad_presentes}b") for i in range(total_estados)
        ]
        
        # Generar etiquetas para las filas (por ejemplo "NCubo 0", "NCubo 1", etc.)
        etiquetas_filas = [f"NCubo {i}" for i in range(tabla.shape[0])]
        
        # Crear DataFrame
        df = pd.DataFrame(tabla, index=etiquetas_filas, columns=etiquetas_columnas)
        print(df.to_string())

    def mostrar_tabla_costos_invertida(self, tabla: np.ndarray, mecanismo_str: str):
        """
        Muestra la tabla de costos con estados binarios como filas y NCubos como columnas.
        
        Parámetros:
        - tabla: np.ndarray de forma (n_ncubos, total_estados)
        - mecanismo_str: str, como "10011"
        """
        cantidad_presentes = sum(bit == "1" for bit in mecanismo_str)
        total_estados = 2 ** cantidad_presentes

        # Etiquetas de filas: estados binarios (00000, 00001, ..., 11111)
        etiquetas_filas = [
            format(i, f"0{cantidad_presentes}b") for i in range(total_estados)
        ]
        
        # Etiquetas de columnas: NCubo 0, NCubo 1, ...
        etiquetas_columnas = [f"NCubo {i}" for i in range(tabla.shape[0])]
        
        # Transponer para que los estados sean las filas
        df = pd.DataFrame(tabla.T, index=etiquetas_filas, columns=etiquetas_columnas)
        
        print(df.to_string())
    
    def vecinos_optimos():
        pass


