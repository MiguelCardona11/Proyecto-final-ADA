from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.main import *
import multiprocessing


def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True

    aplicacion.pagina_sample_network = "C"
    
    iniciar_geometric_individual()
    # iniciar_force_individual()

    # iniciar_estrategia(15, 2, "15A_Qnodos.xlsx")
    # iniciar_qnodes_individual()
    # iniciar_phi_individual()
    
    # Generar la red con 25 elementoss
    # estado_inicio = "1000000000000000000000000"
    # manejador = Manager(estado_inicial=estado_inicio)
    # nombre_archivo = manejador.generar_red(dimensiones=25, datos_discretos=True)
    # print(f"Red generada y guardada en: {nombre_archivo}")
    
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Recomendado en Windows
    main()
