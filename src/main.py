from src.controllers.manager  import Manager
from src.controllers.strategies.phi import Phi
from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.geometric import GeometricSIA
import time
import pandas as pd
import re
import multiprocessing

# Variables para utilizar con NODOS
# 10 NODOS: ABCDEFGHIJ
estado_inicio_10       = "1000000000"
condiciones_10         = "1111111111"
mecanismo_cincuenta_10 = "0111111111"
alcance_cincuenta_10   = "0111111001"
patrones_10            = ["1111111111", "1111111110", "0111111111", "0111111110", "1010101010", "0101010101", "1101101101"]

# 15 NODOS: ABCDEFGHIJKLMNO
estado_inicio_15       = "100000000000000"
condiciones_15         = "111111111111111"
mecanismo_cincuenta_15 = "011111111111111"
alcance_cincuenta_15   = "011111100111111"
patrones_15            = ["111111111111111", "111111111111110", "011111111111111", "011111111111110", "101010101010101", "010101010101010", "110110110110110"]

# 20 NODOS: ABCDEFGHIJKLMNOPQRST
estado_inicio_20       = "10000000000000000000"
condiciones_20         = "11111111111111111111"
mecanismo_cincuenta_20 = "10111111111111111111"
alcance_cincuenta_20   = "10111111111111111111"
patrones_20            = ["11111111111111111111", "11111111111111111110", "01111111111111111111", "01111111111111111110", "10101010101010101010", "01010101010101010101", "11011011011011011011"]

# 25 NODOS: ABCDEFGHIJKLMNOPQRSTUVWXY
estado_inicio_25       = "1000000000000000000000000"
condiciones_25         = "1111111111111111111111111"
mecanismo_cincuenta_25 = "1011111111111111111111111"
alcance_cincuenta_25   = "1011111111111111111111111"
patrones_25            = ["1111111111111111111111111", "1111111111111111111111110", "0111111111111111111111111", "0111111111111111111111110", "1010101010101010101010101", "0101010101010101010101010", "1101101101101101101101101"]

# Resultados de las pruebas
resultados = []

def worker(estado_inicio, condiciones, mecanismo, alcance, estrategia, queue):
    """
    Función que se ejecuta en un proceso separado.
    Se encarga de instanciar el analizador y aplicar la estrategia.
    En lugar de enviar el objeto resultado completo (que puede no ser picklable),
    se extraen sólo los campos necesarios y se envían en un diccionario.
    """
    try:
        config_sistema = Manager(estado_inicial=estado_inicio)
        if estrategia == 1:
            analizador_fi = Phi(config_sistema)
        elif estrategia == 2:
            analizador_fi = QNodes(config_sistema)
        else:
            raise ValueError("Estrategia inválida")
        resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
        # Extraer sólo datos básicos
        result_dict = {
            "perdida": resultado.perdida,
            "estrategia": resultado.estrategia,
            "particion": resultado.particion
        }
        queue.put((True, result_dict))
    except Exception as e:
        # Se envía la cadena de error para asegurarse de que es picklable.
        queue.put((False, str(e)))

def procesar_estrategia(estado_inicio, condiciones, mecanismo, alcance, prueba_num, estrategia: int):
    """
    Ejecuta la prueba en un proceso separado. Si el proceso tarda más de 10800 segundos,
    se termina y se retorna un diccionario con campos en blanco (pero conservando el número de prueba).
    """
    inicio = time.time()
    queue = multiprocessing.Queue()
    proceso = multiprocessing.Process(
        target=worker, 
        args=(estado_inicio, condiciones, mecanismo, alcance, estrategia, queue)
    )
    proceso.start()
    proceso.join(timeout=10800)  # Espera hasta 10800 segundos

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        print(f"La prueba {prueba_num} excedió el límite de tiempo de 10800 segundos y será omitida.")
        return {
            "Prueba": prueba_num,
            "Mecanismo": "",
            "Alcance": "",
            "Pérdida": "",
            "Tiempo": "",
            "Estrategia": "",
            "Partición óptima": ""
        }
    else:
        fin = time.time()
        if not queue.empty():
            exito, resultado = queue.get()
            if exito:
                return {
                    "Prueba": prueba_num,
                    "Mecanismo": mecanismo,
                    "Alcance": alcance,
                    "Pérdida": resultado["perdida"],
                    "Tiempo": round(fin - inicio, 6),
                    "Estrategia": resultado["estrategia"],
                    "Partición óptima": resultado["particion"]
                }
            # else:
            #     print(f"Error en la prueba {prueba_num}: {resultado}")
            #     return {
            #         "Prueba": prueba_num,
            #         "Mecanismo": "",
            #         "Alcance": "",
            #         "Pérdida": "",
            #         "Tiempo": round(fin - inicio, 6),
            #         "Estrategia": "",
            #         "Partición óptima": ""
            #     }
        else:
            print(f"La prueba {prueba_num} terminó sin devolver resultado.")
            return {
                "Prueba": prueba_num,
                "Mecanismo": "",
                "Alcance": "",
                "Pérdida": "",
                "Tiempo": round(fin - inicio, 6),
                "Estrategia": "",
                "Partición óptima": ""
            }

def iniciar_estrategia(cantidad_nodos: int, estrategia: int, nombre_archivo):
    """
    Ejecuta las 50 pruebas de una estrategia dada. 1 --> (Pyphi) 2 --> (QNodes)
    """
    if estrategia not in (1, 2):
        print("Estrategia no soportada, solo se soportan 1 (Pyphi) y 2 (QNodes).")
        return

    if cantidad_nodos == 10:
        estado_inicio = estado_inicio_10
        condiciones = condiciones_10
        mecanismo_cincuenta = mecanismo_cincuenta_10
        alcance_cincuenta = alcance_cincuenta_10
        patrones = patrones_10
    elif cantidad_nodos == 15:
        estado_inicio = estado_inicio_15
        condiciones = condiciones_15
        mecanismo_cincuenta = mecanismo_cincuenta_15
        alcance_cincuenta = alcance_cincuenta_15
        patrones = patrones_15
    elif cantidad_nodos == 20:
        estado_inicio = estado_inicio_20
        condiciones = condiciones_20
        mecanismo_cincuenta = mecanismo_cincuenta_20
        alcance_cincuenta = alcance_cincuenta_20
        patrones = patrones_20
    elif cantidad_nodos == 25:
        estado_inicio = estado_inicio_25
        condiciones = condiciones_25
        mecanismo_cincuenta = mecanismo_cincuenta_25
        alcance_cincuenta = alcance_cincuenta_25
        patrones = patrones_25
    else:
        print("Cantidad de nodos no soportada, solo se soportan 10, 15, 20 y 25 nodos.")
        return

    # Ejecutar las primeras 49 pruebas (7 patrones para alcance x 7 para mecanismo)
    prueba_num = 1
    for i in range(7):
        alcance = patrones[i]
        for mecanismo in patrones:
            resultado = procesar_estrategia(estado_inicio, condiciones, mecanismo, alcance, prueba_num, estrategia)
            resultados.append(resultado)
            print(f"{'Prueba':<8}{'Mecanismo':<30}{'Alcance':<30}{'Pérdida':<30}{'Tiempo(seg)':<30}{'Estrategia'}")
            print("-" * 80)
            print(f"{resultado['Prueba']:<8}{resultado['Mecanismo']:<30}{resultado['Alcance']:<30}{resultado['Pérdida']:<30}{resultado['Tiempo']:<30}{resultado['Estrategia']}")
            print(f"Partición Óptima: \n{resultado['Partición óptima']}\n")
            prueba_num += 1


    # Prueba 50 (caso especial)
    resultado = procesar_estrategia(estado_inicio, condiciones, mecanismo_cincuenta, alcance_cincuenta, prueba_num, estrategia)
    resultados.append(resultado)
    print(f"{'Prueba':<8}{'Mecanismo':<30}{'Alcance':<30}{'Pérdida':<30}{'Tiempo(seg)':<30}{'Estrategia'}")
    print("-" * 80)
    print(f"{resultado['Prueba']:<8}{resultado['Mecanismo']:<30}{resultado['Alcance']:<30}{resultado['Pérdida']:<30}{resultado['Tiempo']:<30}{resultado['Estrategia']}")
    print(f"Partición Óptima: \n{resultado['Partición óptima']}\n")

    # Extraer datos para el Excel
    datos = []
    num = 1
    tipo = ""
    resultado_str = ""
    for obj in resultados:
        entrada = obj['Partición óptima']
        letra_central = re.search(r"[⎛⎝] ([a-zA-Z]) [⎞⎠]", entrada)
        if letra_central:
            letra_central = letra_central.group(1)
            tipo = "mayúscula" if letra_central.isupper() else "minúscula"
        letras_mayusculas = "".join(re.findall(r'[A-Z]', entrada))
        letras_minusculas = "".join(re.findall(r'[a-z]', entrada))
        if tipo == "mayúscula":
            letras_mayusculas = letras_mayusculas[1:]
            resultado_str = f"({letra_central}_t+1|∅)({letras_mayusculas}_t+1|{letras_minusculas}_t)"
        if tipo == "minúscula":
            letras_minusculas = letras_minusculas[1:]
            resultado_str = f"(∅|{letra_central}_t)({letras_mayusculas}_t+1|{letras_minusculas}_t)"
        datos.append({
            "Prueba": num,
            "Particion": resultado_str,
            "Perdida": obj['Pérdida'],
            "Tiempo": obj['Tiempo'],
            "Estrategia": obj['Estrategia']
        })
        num += 1
    
    df = pd.DataFrame(datos)
    df.to_excel(nombre_archivo, index=False, engine="openpyxl")
    print("Archivo Excel generado exitosamente.")
    resultados.clear()

########## PRUEBAS INDIVIDUALES ##########

def iniciar_phi_individual():
    """Ejecución individual de la estrategia Pyphi."""
    estado_inicio =       "0000000000"
    condiciones   =       "1111111111"
    alcance       =       "1111111111"
    mecanismo     =       "1111111111"

    config_sistema = Manager(estado_inicial=estado_inicio)
    analizador_fi = Phi(config_sistema)
    inicio = time.time()
    resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    fin = time.time()
    print(f"  - Pérdida: {resultado.perdida}")
    print(f"  - Tiempo: {fin - inicio:.6f} s")
    print(f"  - Partición óptima: \n{resultado.particion}")
    
def iniciar_force_individual():
    """Ejecución individual de la estrategia Bruteforce."""
    estado_inicio =       "00000"
    condiciones   =       "11111"
    alcance       =       "11111"
    mecanismo     =       "11111"

    config_sistema = Manager(estado_inicial=estado_inicio)
    analizador_fi = BruteForce(config_sistema)
    inicio = time.time()
    resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    fin = time.time()
    print(f"  - Pérdida: {resultado.perdida}")
    print(f"  - Tiempo: {fin - inicio:.6f} s")
    print(f"  - Partición óptima: \n{resultado.particion}")


def iniciar_qnodes_individual():
    """Ejecución individual de la estrategia QNodes."""
    estado_inicio =       "000"
    condiciones   =       "111"
    alcance       =       "111"
    mecanismo     =       "111"

    config_sistema = Manager(estado_inicial=estado_inicio)
    analizador_fi = QNodes(config_sistema)
    inicio = time.time()
    resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    fin = time.time()
    print(f"  - Pérdida: {resultado.perdida}")
    print(f"  - Tiempo: {fin - inicio:.6f} s")
    print(f"  - Partición óptima: \n{resultado.particion}")
    
def iniciar_geometric_individual():
    """Ejecución individual de la estrategia Geometric."""
    estado_inicio =       "000"
    condiciones   =       "111"
    alcance       =       "111"
    mecanismo     =       "111"

    config_sistema = Manager(estado_inicial=estado_inicio)
    analizador_fi = GeometricSIA(config_sistema)
    resultado = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo)
    print(resultado)