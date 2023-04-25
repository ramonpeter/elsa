""" Define LHC preprocess module """

from typing import Union
import torch
import numpy as np
from ..mappings.rambo import RamboOnDietHadron
from abc import ABC, abstractmethod
import sys


class Scaler():
    
    def __init__(self, is_hypercube: bool):
        self.is_hypercube = is_hypercube
        self.is_fitted = False
       
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """Needs to be implemented by child class"""
        raise NotImplementedError()
    
    def fit(self, x: np.ndarray):
        """Fit the scalings params"""
        if not self.fitted:
            x = self._reparam(x)
            if self.is_hypercube:
                self.mean = self.feature_min
                self.scale = self.feature_max - self.feature_min
            else:
                self.mean = x.mean(axis=0, keepdims=True)
                self.scale = x.std(axis=0, keepdims=True)
                
            self.fitted = True
            
    def fit_and_transform(self, x: np.ndarray):
        """Fits and transforms"""
        self.fit(x)
        z = self.transform(x)
        return z
        
    def transform(self, x: np.ndarray):
        """Forward transformation"""
        if not self.fitted:
            raise ValueError("Not fitted yet")
        z = self._reparam(x)
        z -= self.mean
        z /= self.scale
        return z
        
    def inverse_transform(self, z: np.ndarray):
        """Inverse transformation"""
        if not self.fitted:
            raise ValueError("Not fitted yet")
        z *= self.scale
        z += self.mean
        x = self._reparam(z, inverse=True)
        return x


class SimpleScaler(Scaler):
    """Basic preprocessing. Just scaling the input
    with a given mean and scale given as input"""
    
    def __init__(self, scale: np.ndarray, mean: np.ndarray=None):
        """
        Args:
            scale (np.ndarray): scaling variable
            mean (np.ndarray, optional): mean variable. Defaults to None.
        """
        super().__init__(False)
        self.mean = mean
        self.scale = scale       
        self.fitted = True
        
        self.feature_min = None
        self.feature_max = None
        
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """No reparametrization"""
        return x_or_z
        
    
class SchumannScaler(Scaler):
    """Propreccing of LHC data
    Uses the preprocessing suggested in arxiv:2109.11964.
    
    Takes 3 momenta of initial and final states, resulting in

    d = 3n + 2,
    
    dimensions, as the pT of the initial states vanish. Final
    output is mapped onto [0, 1]. The Ref. used [-1,1] but
    this should not change anything.
    """
    def __init__(self, e_had: float, n_particles: int, masses: list = None, is_hypercube: bool=False, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube)
        self.e_had = e_had
        self.masses = np.array(masses)[None,...]
        self.n_particles = n_particles
        
        self.fitted = False
        self.feature_min = -e_had/2
        self.feature_max = e_had/2
    
    def _reparam(self, x_or_z: np.ndarray, inverse=False):
        """
        Makes reparametrization into
            x = {pf1,..,pfn, pi1, pi2}
        with
            pf = {px, py, pz} and pi = {pz}
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles + 2)
       
        """
        out = []
        aug = []
        
        if inverse:
            #--- Inverse direction ----#
            x = x_or_z[:, :3*self.n_particles]
            
            # x.shape:     (b, 3 * n_particles)
            # reshape   -> (b, n_particles, 3)
            # transpose -> (b, 3, n_particles)
            x = np.reshape(x, (x.shape[0], self.n_particles, 3)).transpose(0, 2, 1)
            
            # get Px, Py, Pz
            Xs = x[:, 0, :]
            Ys = x[:, 1, :]
            Zs = x[:, 2, :]
            Es = np.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
            
            out.append(Es)
            out.append(Xs)
            out.append(Ys)
            out.append(Zs)
            x_out = np.stack(out, axis=1)
            
            # x_out.shape: (b, 4, n_particles)
            # transpose -> (b, n_particles, n_out)
            # reshape   -> (b, 4 * n_particles)
            shape_dim = 4 * self.n_particles
            x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
            return x_out

        #--- Forward direction ----#
        
        # x.shape:     (b, 4 * n_particles)
        # reshape   -> (b, n_particles, 4)
        # transpose -> (b, 4, n_particles)
        x = np.reshape(x_or_z, (x_or_z.shape[0], self.n_particles, 4)).transpose(0, 2, 1)

        # get E, Px, Py, Pz
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]
        out.append(Xs)
        out.append(Ys)
        out.append(Zs)
        y = np.stack(out, axis=1)
        
        # y.shape:     (b, 3, n_particles)
        # transpose -> (b, n_particles, 3)
        # reshape   -> (b, 3 * n_particles)
        shape_dim = 3 * self.n_particles
        y = np.transpose(y, (0, 2, 1)).reshape(y.shape[0], shape_dim)
        
        # shape  -> (b, 1)
        Et  = np.sum(Es, axis=1, keepdims=True)
        Pzt = np.sum(Zs, axis=1, keepdims=True)
        Pz1 = (Pzt + Et)/2
        Pz2 = (Pzt - Et)/2
        aug.append(Pz1)
        aug.append(Pz2)
        x_aug = np.concatenate(aug, axis=-1)
        
        # Concat all entries
        x_out = np.concatenate([y, x_aug], axis=-1)
        
        return x_out

    
class ThreeMomScaler(Scaler):
    """Propreccing of LHC data
    Only takes 3 momenta of final states, resulting in

    d = 3n,
    
    dimensions and features. Final output is mapped onto [0,1] 
    , or centered and scaled (is_hypercube=False).
    """

    def __init__(self, e_had: float, n_particles: int, masses: list = None, is_hypercube: bool=False, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube)
        self.e_had = e_had
        self.masses = np.array(masses)[None,...]
        self.n_particles = n_particles
        
        self.fitted = False
        self.feature_min = -e_had/2
        self.feature_max = e_had/2
    
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """
        Makes reparametrization into
            x = {pf1,..,pfn}
        with
            pf = {px, py, pz}
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles)

        """
        if inverse:
            #--- Inverse direction ----#
            out = []
            # x.shape:     (b, 3 * n_particles )
            # reshape   -> (b, n_particles , 3)
            # transpose -> (b, 3, n_particles)
            x = np.reshape(x_or_z, (x_or_z.shape[0], self.n_particles , 3)).transpose(0, 2, 1)
            
            # get Px, Py, Pz
            Xs = x[:, 0, :]
            Ys = x[:, 1, :]
            Zs = x[:, 2, :]
            Es = np.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
            
            out.append(Es)
            out.append(Xs)
            out.append(Ys)
            out.append(Zs)
            x_out = np.stack(out, axis=1)
            
            # x_out.shape: (b, 4, n_particles)
            # transpose -> (b, n_particles, n_out)
            # reshape   -> (b, 4 * n_particles)
            shape_dim = 4 * self.n_particles
            x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
            return x_out
        
        #--- Forward direction ----#
        mask = [False, True, True, True] * self.n_particles
        z = x_or_z[:,mask]
        return z

    
class MinimalRepScaler(Scaler):
    """Propreccing of LHC data
    Only takes a minimal rep, using only

    d = 3n - 2,
    
    dimensions and features. Final output is mapped onto [0,1] 
    , or centered and scaled (is_hypercube=False).
    """

    def __init__(self, e_had: float, n_particles: int, masses: list = None, is_hypercube: bool=True, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube)
        self.e_had = e_had
        self.masses = np.array(masses)[None,...]
        self.n_particles = n_particles
       
        self.fitted = False
        self.feature_min = -e_had/2
        self.feature_max = e_had/2
    
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """
        Makes reparametrization into
            x = {pf1,..,pfn}
        with
            pf = {px, py, pz} and only pfn={pz}
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles - 2)

        """
        if inverse:
            #--- Inverse direction ----#
            out = []
            x, Zlast = x_or_z[:,:3*(self.n_particles-1)], x_or_z[:,-1:]
            
            # x.shape:     (b, 3 * (n_particles - 1))
            # reshape   -> (b, n_particles - 1, 3)
            # transpose -> (b, 3, n_particles - 1)
            x = np.reshape(x, (x.shape[0], self.n_particles - 1, 3)).transpose(0, 2, 1)
            Xlast = -np.sum(x[:, 0, :], axis=-1, keepdims=True)
            Ylast = -np.sum(x[:, 1, :], axis=-1, keepdims=True)
            
            # get Px, Py, Pz
            Xs = np.concatenate([x[:, 0, :], Xlast], axis=-1)
            Ys = np.concatenate([x[:, 1, :], Ylast], axis=-1)
            Zs = np.concatenate([x[:, 2, :], Zlast], axis=-1)
            Es = np.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
            
            out.append(Es)
            out.append(Xs)
            out.append(Ys)
            out.append(Zs)
            x_out = np.stack(out, axis=1)
            
            # x_out.shape: (b, 4, n_particles)
            # transpose -> (b, n_particles, n_out)
            # reshape   -> (b, 4 * n_particles)
            shape_dim = 4 * self.n_particles
            x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
            
            return x_out
        
        #--- Forward direction ----#
        mask = [False, True, True, True] * (self.n_particles - 1) + [False, False, False, True]
        z = x_or_z[:,mask]
        return z
    
class HeimelScaler(Scaler):
    """Propreccing of LHC data
    Uses the preprocessing suggested in arxiv:2110.13632.
    Takes transformed 3-mom representation of final state particles yielding
    
    d = 3n,
    
    dimensions and features. Final output is mapped onto [0,1] 
    , or centered and scaled (is_hypercube=False = Default).
    """

    def __init__(
        self, 
        e_had: float, 
        n_particles: int, 
        masses: list, 
        ptcuts: list, 
        is_hypercube: bool=False,
        **kwargs
    ):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube)
        self.e_had = e_had
        self.masses = np.array(masses)[None,...]
        self.ptcuts = np.array(ptcuts)[None,...]
        assert self.masses.shape[1] == self.ptcuts.shape[1]
        self.n_particles = n_particles
       
        self.fitted = False
        self.feature_min = -e_had/2
        self.feature_max = e_had/2
    
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """
        Makes reparametrization into
            x = {pf1,..,pfn}
        with
            pf = {px, py, pz} and only pfn={pz}
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles - 2)

        """
        if inverse:
            #--- Inverse direction ----#
            out = []
            # x.shape:     (b, 3 * n_particles )
            # reshape   -> (b, n_particles, 3)
            # transpose -> (b, 3, n_particles)
            x = np.reshape(x_or_z, (x_or_z.shape[0], self.n_particles, 3)).transpose(0, 2, 1)
            
            # get Pt, Eta, Phi transformed
            Pt = x[:, 0, :]
            Eta = x[:, 1, :]
            Delta_phi = x[:, 2, :]
            Phi = np.zeros_like(Delta_phi)
            
            # Get proper Pt and phi and eta
            Pt = np.exp(Pt) + self.ptcuts
            Delta_phi = np.tanh(Delta_phi) * np.pi
            Phi = (Delta_phi - np.pi + Delta_phi[:,:1]) % (-2*np.pi) + np.pi
            Phi[:,0] = Delta_phi[:,0]
            
            # get Px, Py, Pz
            Xs = Pt * np.cos(Phi)
            Ys = Pt * np.sin(Phi)
            Zs = Pt * np.sinh(Eta)
            Es = np.sqrt(self.masses**2 + Xs ** 2 + Ys ** 2 + Zs ** 2)
            
            out.append(Es)
            out.append(Xs)
            out.append(Ys)
            out.append(Zs)
            x_out = np.stack(out, axis=1)
            
            # x_out.shape: (b, 4, n_particles)
            # transpose -> (b, n_particles, n_out)
            # reshape   -> (b, 4 * n_particles)
            shape_dim = 4 * self.n_particles
            x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
            return x_out
        
        #--- Forward direction ----#
        out = []
        # x.shape:     (b, 4 * n_particles)
        # reshape   -> (b, n_particles, 4)
        # transpose -> (b, 4, n_particles)
        x = np.reshape(x_or_z, (x_or_z.shape[0], self.n_particles, 4)).transpose(0, 2, 1)
        
        # get E, Px, Py, Pz
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]
        
        # get pt, eta, phi
        Pp = np.sqrt(Xs **2 + Ys**2 + Zs**2)
        Pt = np.sqrt(Xs **2 + Ys**2)
        Eta = np.arctanh(Zs/Pp)
        Phi = np.arctan2(Ys,Xs)
        
        # Get transformed pt
        Pt = np.log(Pt - self.ptcuts)
        
        # Get transformed delta phis
        Delta_phi = (Phi - Phi[:,:1] + np.pi) % (2*np.pi) - np.pi
        Delta_phi[:,0] = Phi[:,0]
        Delta_phi = np.arctanh(Delta_phi/np.pi)
        
        # Add output
        out.append(Pt)
        out.append(Eta)
        out.append(Delta_phi)
        x_out = np.stack(out, axis=1)
        
        # x_out.shape: (b, 3, n_particles)
        # transpose -> (b, n_particles, 3)
        # reshape   -> (b, 3 * n_particles)
        shape_dim = 3 * self.n_particles
        x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
        
        return x_out
    
class LaserScaler(Scaler):
    """Propreccing of LHC data
    Takes transformed 3-mom representation of final state particles including
    some augmented features, yielding
    
    d = 3n + m,
    
    dimensions and features. Final output is mapped onto [0,1] 
    , or centered and scaled (is_hypercube=False = Default).
    """

    def __init__(
        self, 
        e_had: float, 
        n_particles: int, 
        masses: list, 
        ptcuts: list, 
        is_hypercube: bool=False,
        **kwargs
    ):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube)
        self.e_had = e_had
        self.masses = np.array(masses)[None,...]
        self.ptcuts = np.array(ptcuts)[None,...]
        assert self.masses.shape[1] == self.ptcuts.shape[1]
        self.n_particles = n_particles
       
        self.fitted = False
        self.feature_min = -e_had/2
        self.feature_max = e_had/2
    
    def _reparam(self, x_or_z: np.ndarray, inverse: bool=False):
        """
        Makes reparametrization into
            x = {pf1,..,pfn}
        with
            pf = {px, py, pz} and only pfn={pz}
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles - 2)

        """
        if inverse:
            raise ValueError("No inverse available!! Only use it for the discriminator")
        
        #--- Forward direction ----#
        out = []
        aug = []
        # x.shape:     (b, 4 * n_particles)
        # reshape   -> (b, n_particles, 4)
        # transpose -> (b, 4, n_particles)
        x = np.reshape(x_or_z, (x_or_z.shape[0], self.n_particles, 4)).transpose(0, 2, 1)
        
        # get E, Px, Py, Pz
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]
        
        # get pt, eta, phi
        Pp = np.sqrt(Xs **2 + Ys**2 + Zs**2)
        Pt = np.sqrt(Xs **2 + Ys**2)
        Eta = np.arctanh(Zs/Pp)
        Phi = np.arctan2(Ys,Xs)
        
        # Get transformed pt
        Pt = np.log(Pt)
        
        # # Add output
        out.append(Pt)
        out.append(Eta)
        
        # Get transformed delta phis
        for i in range(1, self.n_particles):
            for j in range(i+1, self.n_particles):
                dphi = np.fabs(Phi[:,[i]] - Phi[:,[j]])
                dphimin = np.where(dphi>np.pi, 2.0 * np.pi - dphi, dphi)
                dphimin = np.arctanh(dphimin/np.pi * 2 - 1)
                aug.append(dphimin)
                
        # Get transformed delta etas
        for i in range(1, self.n_particles):
            for j in range(i+1, self.n_particles):
                deta = np.abs(Eta[:,[i]] - Eta[:,[j]])
                aug.append(np.log(deta))
                
        # Get transformed delta R
        for i in range(1, self.n_particles):
            for j in range(i+1, self.n_particles):
                dphi = np.fabs(Phi[:,[i]] - Phi[:,[j]])
                deta = np.abs(Eta[:,[i]] - Eta[:,[j]])
                dphimin = np.where(dphi>np.pi, 2.0 * np.pi - dphi, dphi)
                dR = np.sqrt(dphimin ** 2 + deta ** 2)
                aug.append(np.log(dR))
    
        n_out = len(out)
        x_out = np.stack(out, axis=1)
        
        # x_out.shape: (b, features, n_particles)
        # transpose -> (b, n_particles, features)
        # reshape   -> (b, features * n_particles)
        shape_dim =  n_out * self.n_particles
        x_out = np.transpose(x_out, (0, 2, 1)).reshape(x_out.shape[0], shape_dim)
        
        x_aug = np.concatenate(aug, axis=-1)
        return np.concatenate([x_out, x_aug], axis=-1)
    
class RamboScaler(Scaler):
    """Propreccing of LHC data"""

    def __init__(self, e_had: float, n_particles: int, masses: list = None, **kwargs):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(is_hypercube=True)
        self.ps_mapping = RamboOnDietHadron(e_had, n_particles, masses)

        self.fitted = True
        self.feature_min = 0
        self.feature_max = 1
        self.mean = 0
        self.scale = 1
        
    def _reparam(self, x_or_z: np.ndarray, inverse=False):
        """
        Performs mapping onto unit-hypercube.
        According to rambo algorithm
        
        And transforms the shapes according to:
            (batch_size, 4 * n_particles) 
                <--inverse | forward--> 
            (batch_size, 3 * n_particles + 2)
       
        """      
        if inverse:
            #--- Inverse direction ----#
            z = self.ps_mapping.map(x_or_z)
            return z

        #--- Forward direction ----#
        x = self.ps_mapping.map_inverse(x_or_z)
        
        return x
