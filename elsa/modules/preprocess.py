""" Define LHC preprocess module """

from typing import Union
import torch
import numpy as np
from ..mappings.rambo import RamboOnDietHadron
from abc import ABC, abstractmethod
import sys


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
        self.fitted = True
        
    def fit_and_transform(self, x):
        return self.transform(x)

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
    
class SherpaScaler(Scaler):
    """Propreccing of LHC data
    Uses the preprocessing suggested in arxiv:2109.11964.
    
    Takes 3 momenta of initial and final states, resulting in

    d = 3n + 2,
    
    dimensions, as the pT of the initial states vanish. Final
    output is mapped onto [0, 1]. The Ref. used [-1,1] but
    this should not change anything.
    """
    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        self.e_had = e_had
        self.masses = torch.Tensor(masses)[None,...]
        self.n_particles = n_particles
        self.fitted = False
        self.hypercube = True
        
    def fit_and_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def fit(self, x):
        x = self._reparam(x)
        self.mean = -self.e_had/2
        self.scale = self.e_had
        self.fitted = True
    
    def _reparam(self, x):
        
        out = []
        aug = []
        x = torch.tensor(x)
        
        # x.shape:     (b, 4 * n_particles)
        # reshape   -> (b, n_particles, 4)
        # transpose -> (b, 4, n_particles)
        x = torch.reshape(x, (x.shape[0], self.n_particles, 4)).transpose(1, 2)

        # get E, Px, Py, Pz
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]
        out.append(Xs)
        out.append(Ys)
        out.append(Zs)
        y = torch.stack(out, dim=1)
        
        # y.shape:     (b, 3, n_particles)
        # transpose -> (b, n_particles, 3)
        # reshape   -> (b, 3 * n_particles)
        shape_dim = 3 * self.n_particles
        y = torch.transpose(y, 1, 2).reshape(y.shape[0], shape_dim)
        
        # shape  -> (b, 1)
        Et  = torch.sum(Es, dim=1, keepdim=True)
        Pzt = torch.sum(Zs, dim=1, keepdim=True)
        Pz1 = (Pzt + Et)/2
        Pz2 = (Pzt - Et)/2
        aug.append(Pz1)
        aug.append(Pz2)
        x_aug = torch.cat(aug, dim=-1)
        
        # Concat all entries
        x_out = torch.cat([y, x_aug], dim=-1)
        
        return x_out
        

    def transform(self, x):
        """
        Maps momentum features into [0,1]

        Args:
            x: Tensor with shape (batch_size, 4 * n_particles).

        Returns:
            z: Tensor with shape (batch_size, 3 * n_particles + 2).
        """
        x_out = self._reparam(x)
        x_out -= self.mean
        x_out /= self.scale

        return x_out.numpy()

    def inverse_transform(self, z):
        """
        Maps back to the original momentum representation.

        Args:
            z: Tensor with shape (batch_size, 3 * n_particles + 2).

        Returns:
            x: Tensor with shape (batch_size, 4 * n_particles).
        """
        out = []
        z = torch.tensor(z)
        
        z *= self.scale
        z += self.mean
        x, _ = torch.split(z, [3*self.n_particles, 2], dim=-1)
        
        # x.shape:     (b, 3 * n_particles)
        # reshape   -> (b, n_particles, 3)
        # transpose -> (b, 3, n_particles)
        x = torch.reshape(x, (x.shape[0], self.n_particles, 3)).transpose(1, 2)
        
        # get Px, Py, Pz
        Xs = x[:, 0, :]
        Ys = x[:, 1, :]
        Zs = x[:, 2, :]
        Es = torch.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
        
        out.append(Es)
        out.append(Xs)
        out.append(Ys)
        out.append(Zs)
        x_out = torch.stack(out, dim=1)
        
        # x_out.shape: (b, 4, n_particles)
        # transpose -> (b, n_particles, n_out)
        # reshape   -> (b, 4 * n_particles)
        shape_dim = 4 * self.n_particles
        x_out = torch.transpose(x_out, 1, 2).reshape(x_out.shape[0], shape_dim)
        return x_out.numpy()
    
class ThreeMomScaler(Scaler):
    """Propreccing of LHC data"""

    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        self.e_had = e_had
        self.masses = torch.Tensor(masses)[None,...]
        self.n_particles = n_particles

    def transform(self, x):
        """
        Maps momentum features between [-1, 1]

        Args:
            x: Tensor with shape (batch_size, 4 * n_particles).

        Returns:
            z: Tensor with shape (batch_size, 3 * n_particles).
        """
        mask = [False, True, True, True] * self.n_particles
        z = x[:,mask]
        z += self.e_had/2
        z /= self.e_had

        return z

    def inverse_transform(self, z):
        """
        Maps back to the original momentum representation.

        Args:
            z: Tensor with shape (batch_size, 3 * n_particles).

        Returns:
            x: Tensor with shape (batch_size, 4 * n_particles).
        """
        out = []
        z *= self.e_had
        z -= self.e_had/2
        
        # x.shape:     (b, 3 * n_particles )
        # reshape   -> (b, n_particles , 3)
        # transpose -> (b, 3, n_particles)
        x = torch.reshape(x, (x.shape()[0], self.n_particles , 3)).transpose(1, 2)
        
        # get Px, Py, Pz
        Xs = x[:, 0, :]
        Ys = x[:, 1, :]
        Zs = x[:, 2, :]
        Es = torch.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
        
        out.append(Es)
        out.append(Xs)
        out.append(Ys)
        out.append(Zs)
        x_out = torch.stack(out, dim=1)
        
        # x_out.shape: (b, 4, n_particles)
        # transpose -> (b, n_particles, n_out)
        # reshape   -> (b, 4 * n_particles)
        shape_dim = 4 * self.n_particles
        x_out = torch.transpose(x_out, 1, 2).reshape(x_out.shape[0], shape_dim)
        
        return x
    
class MinimalRepScaler(Scaler):
    """Propreccing of LHC data"""

    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        self.e_had = e_had
        self.masses = torch.Tensor(masses)[None,...]
        self.n_particles = n_particles
        self.fitted = False
        
    def fit_and_transform(self, x):
        self.fit(x)
        return self.transform(x)
        
    def fit(self, x):
        x = torch.tensor(x)
        
        mask = [False, True, True, True] * (self.n_particles - 1) + [False, False, False, True]
        z = x[:,mask]

        self.mean = z.mean(axis=0, keepdim=True)
        self.std = z.std(axis=0, keepdim=True)
        self.fitted = True

    def transform(self, x):
        """
        Maps momentum features between [0, 1]

        Args:
            x: Tensor with shape (batch_size, 4 * n_particles).

        Returns:
            z: Tensor with shape (batch_size, 3 * n_particles - 2).
        """
        if not self.fitted:
            raise ValueError("Not fitted yet")
        
        x = torch.tensor(x)
        
        mask = [False, True, True, True] * (self.n_particles - 1) + [False, False, False, True]
        z = x[:,mask]
        
        z -= self.mean
        z /= self.std

        return z.numpy()

    def inverse_transform(self, z):
        """
        Maps back to the original momentum representation.

        Args:
            z: Tensor with shape (batch_size, 3 * n_particles - 2).

        Returns:
            x: Tensor with shape (batch_size, 4 * n_particles).
        """
        if not self.fitted:
            raise ValueError("Not fitted yet")
        
        out = []
        z = torch.tensor(z)
        
        z *= self.std
        z += self.mean
        x, Zlast = torch.split(z, [3*(self.n_particles-1), 1], dim=-1)
        
        # x.shape:     (b, 3 * (n_particles - 1))
        # reshape   -> (b, n_particles - 1, 3)
        # transpose -> (b, 3, n_particles - 1)
        x = torch.reshape(x, (x.shape[0], self.n_particles - 1, 3)).transpose(1, 2)
        Xlast = -torch.sum(x[:, 0, :], dim=-1, keepdim=True)
        Ylast = -torch.sum(x[:, 1, :], dim=-1, keepdim=True)
        
        # get Px, Py, Pz
        Xs = torch.cat([x[:, 0, :], Xlast], dim=-1)
        Ys = torch.cat([x[:, 1, :], Ylast], dim=-1)
        Zs = torch.cat([x[:, 2, :], Zlast], dim=-1)
        Es = torch.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
        
        out.append(Es)
        out.append(Xs)
        out.append(Ys)
        out.append(Zs)
        x_out = torch.stack(out, dim=1)
        
        # x_out.shape: (b, 4, n_particles)
        # transpose -> (b, n_particles, n_out)
        # reshape   -> (b, 4 * n_particles)
        shape_dim = 4 * self.n_particles
        x_out = torch.transpose(x_out, 1, 2).reshape(x_out.shape[0], shape_dim)
        
        return x_out.numpy()
    
class RamboScaler(Scaler):
    """Propreccing of LHC data"""

    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        self.ps_mapping = RamboOnDietHadron(e_had, n_particles, masses)
        self.fitted = True
        
    def fit_and_transform(self, x):
        return self.transform(x)

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
