import numpy as np
from typing import Union, List, Tuple, Dict

class OLAPCube:
    def __init__(self, num_dims: int):
        """
        Inicializa un hipercubo de 2^num_dims posiciones, donde cada posición almacena una métrica.
        """
        self.num_dims = num_dims
        self.shape = tuple([2] * num_dims)
        self.data = np.zeros(self.shape)

    def set_value(self, index: Tuple[int, ...], value: float):
        """Establece una probabilidad condicional en un índice binario específico."""
        self.data[index] = value

    def get_value(self, index: Tuple[int, ...]) -> float:
        """Obtiene la probabilidad condicional almacenada en un índice binario específico."""
        return self.data[index]

    def hamming(index: Tuple[int, ...], index2: Tuple[int, ...]) -> float:
        if len(index) != len(index2):
            raise ValueError("Las tuplas deben tener la misma longitud.")
        distancia = sum(1 for a, b in zip(index, index2) if a != b)
        return float(distancia)
       



    def slice(self, dim: int, val: int):
        """
        Devuelve un nuevo OLAPCube con una dimensión fija (condicionamiento).
        """
        sliced_data = self.data.take(indices=val, axis=dim)
        new_cube = OLAPCube(self.num_dims - 1)
        new_cube.data = sliced_data
        return new_cube

    def dice(self, constraints: Dict[int, List[int]]):
        """
        Devuelve un subcubo con múltiples restricciones sobre las dimensiones.
        constraints = {dim1: [val1, val2], dim2: [val3], ...}
        """
        mask = np.ones(self.shape, dtype=bool)
        for dim, vals in constraints.items():
            axis_index = np.array([False]*2)
            for v in vals:
                axis_index[v] = True
            slicer = [slice(None)] * self.num_dims
            slicer[dim] = axis_index
            mask &= axis_index[np.newaxis if dim == 0 else slice(None)]
        return self.data[mask]

    def rollup(self, dim: int):
        """
        Agrega (suma) sobre una dimensión, análogo a marginalizar.
        """
        rolled = np.sum(self.data, axis=dim)
        new_cube = OLAPCube(self.num_dims - 1)
        new_cube.data = rolled
        return new_cube

    def drilldown(self, dim: int):
        """
        Expansión detallada de una dimensión agregada — no aplica directamente si ya fue colapsada.
        """
        raise NotImplementedError("Drill-down requiere una jerarquía o expansión previa de roll-up.")

    def __str__(self):
        return f"OLAPCube(dim={self.num_dims})\n{self.data}"
