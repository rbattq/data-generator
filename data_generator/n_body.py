import numpy as np
import pandas as pd
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


def random_conformations(m: int, n: int, rng: np.random.Generator, scale) -> np.ndarray:
    """
    Produce m conformations of n particles
    :param m: number of conformations
    :param n: number of particles
    :param rng: random number generator
    :param scale: scale of conformations
    :return: m x n x 3 dimensional array
    """
    return scale * rng.uniform(-1, 1, size=[m, n, 3])


def coords2dataframe(r: np.ndarray) -> pd.DataFrame:
    d = {}
    for i in range(r.shape[1]):
        for j in range(3):
            letter = "xyz"[j]
            key = f"{letter}{i+1}"
            d[key] = r[:, i, j]
    return pd.DataFrame(d)


@dataclass
class NBody:
    m0: float
    m: List[float]

    @property
    def n(self):
        return len(self.m)

    def total_force(self, r: np.ndarray) -> np.ndarray:
        """
        Computes total force of all other particles on particle 0
        :param r: displacements of other particls from particle 0
        :return: total force on particle 0
        """
        f = np.zeros(3)
        for i in range(self.n):
            f += compute_force(self.m0, self.m[i], r[i])

        return f

    def generate_random_observations(
        self, m: int, rng: np.random.Generator, scale: float, epsilon: float
    ) -> pd.DataFrame:
        """
        Generate random conformations, record perturbed positions and computed magnitude of fource
        :param m: number of observations
        :param scale: scale of random positions
        :param epsilon: scale of observation error of positions
        :return: table of observations where last column is magnitude of force
        """
        r = random_conformations(m, self.n, rng, scale)
        delta = random_conformations(m, self.n, rng, epsilon)
        df = coords2dataframe(r + delta)
        forces = [self.total_force(row) for row in r]
        forces = [np.linalg.norm(f) for f in forces]
        df["forces"] = forces
        return df
