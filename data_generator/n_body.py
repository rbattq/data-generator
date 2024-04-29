import numpy as np
from typing import Dict, Type, List, Tuple
from dataclasses import dataclass, field


def compute_force(m0: float, m1: float, r: np.ndarray) -> np.ndarray:
    """
    Compute gravitational force on particle 0 exerted by particle 1
    :param m0: mass of particle 0
    :param m1: mass of particle 1
    :param r: displacement vector from particle 0 to particle 1
    :return: force vector
    """
    return -(m0 * m1 / np.linalg.norm(r) ** 3) * r


@dataclass
class NBody:
    m0: float
    m: List[float]

    def total_force(self, r: np.ndarray) -> np.ndarray:
        """
        Computes total force of all other particles on particle 0
        :param r: displacements of other particls from particle 0
        :return: total force on particle 0
        """
        f = np.zeros(3)
        for i in range(len(self.m)):
            f += compute_force(self.m0, self.m[i], r[i])

        return f
