""" Define LHC preprocess module """

from typing import Union
import torch
import numpy as np
from ..mappings.rambo import RamboOnDietHadron
from abc import ABC, abstractmethod


class Scaler(ABC):
    
    @abstractmethod
    def transform(self):
        pass
        
    @abstractmethod
    def inverse_transform(self):
        pass


class SimpleScaler(Scaler):
    """Basic preprocessing"""

    def __init__(self, scale: np.ndarray, mean: np.ndarray=None):
        """
        Args:
            scale (np.ndarray): scaling variable
            mean (np.ndarray, optional): mean variable. Defaults to None.
        """
        self.scale = scale
        self.mean = mean

    def transform(self, x):
        """
        Perform standardization by {centering} and scaling.
        
        Args:
            x: Tensor of shape (n_samples, n_features).

        Returns:
            x_tr: Transformed tensor of shape (n_samples, n_features)
        """
        if self.mean:
            x -= self.mean
        x /= self.scale
        return x

    def inverse_transform(self, x):
        """
        Scale back the data to the original representation.

        Args:
            x: Tensor of shape (n_samples, n_features).

        Returns:
            x_tr: Transformed tensor of shape (n_samples, n_features)
        """
        x *= self.scale
        if self.mean:
            x += self.mean
        return x
    
class PhysicsScaler(Scaler):
    """Propreccing of LHC data"""

    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        self.ps_mapping = RamboOnDietHadron(e_had, n_particles, masses)

    def transform(self, x):
        """
        Performs mapping onto unit-hypercube.

        Args:
            x: Tensor with shape (batch_size, 4 * n_particles).

        Returns:
            z: Tensor with shape (batch_size, 3 * n_particles - 2).
        """
        z = self.ps_mapping.map_inverse(x)
        return z

    def inverse_transform(self, z):
        """
        Maps back to the original momentum representation.

        Args:
            z: Tensor with shape (batch_size, 3 * n_particles - 2).

        Returns:
            x: Tensor with shape (batch_size, 4 * n_particles).
        """
        x = self.ps_mapping.map(z)
        return x
