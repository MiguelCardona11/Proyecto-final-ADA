from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.main import *
import multiprocessing


def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True

    aplicacion.pagina_sample_network = "A"
    
    iniciar_geometric_individual()
    # iniciar_estrategia(10, 3, "10A_GEOMETRIC.xlsx")
    # iniciar_qnodes_individual()
    # iniciar_phi_individual()
    
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Recomendado en Windows
    main()