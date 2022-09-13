import numpy as np
from scipy.optimize import brentq
from math import gamma
import sys

from .base import PhaseSpaceMapping
from .helper import interpret_array, map_fourvector_rambo, two_body_decay_factor, boost, boost_z

# Define Metric
MINKOWSKI = np.diag([1, -1, -1, -1])

class Rambo(PhaseSpaceMapping):

    def __init__(self, e_cm, nparticles):
        self.e_cm = e_cm
        self.nparticles = nparticles
        super(Rambo, self).__init__(nparticles * 4, nparticles * 4)

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = ((np.pi / 2.) ** (nparticles - 1) * e_cm ** (2 * nparticles - 4) /
               (gamma(nparticles) * gamma(nparticles - 1)))
        return 1 / vol

    def map(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm

        p = np.empty((xs.shape[0], nparticles, 4))

        q = map_fourvector_rambo(xs.reshape(xs.shape[0], nparticles, 4))
        # sum over all particles
        Q = np.add.reduce(q, axis=1)

        M = np.sqrt(np.einsum('kd,dd,kd->k', Q, MINKOWSKI, Q))
        b = (-Q[:, 1:] / M[:, np.newaxis])
        x = e_cm / M
        gamma = Q[:, 0] / M
        a = 1. / (1. + gamma)

        bdotq = np.einsum('ki,kpi->kp', b, q[:, :, 1:])

        # make dimensions match
        gamma = gamma[:, np.newaxis]
        x = x[:, np.newaxis]
        p[:, :, 0] = x * (gamma * q[:, :, 0] + bdotq)

        # make dimensions match
        b = b[:, np.newaxis, :]  # dimensions: samples * nparticles * space dim)
        bdotq = bdotq[:, :, np.newaxis]
        x = x[:, :, np.newaxis]
        a = a[:, np.newaxis, np.newaxis]
        p[:, :, 1:] = x * (
                    q[:, :, 1:] + b * q[:, :, 0, np.newaxis] + a * bdotq * b)

        return p.reshape(xs.shape)

    def pdf_gradient(self, xs):
        return 0

    def map_inverse(self, xs):
        raise NotImplementedError


class RamboOnDiet(PhaseSpaceMapping):

    def __init__(self, e_cm, nparticles):
        self.e_cm = np.float64(e_cm) # The cast is important for accurate results!
        self.nparticles = nparticles
        super(RamboOnDiet, self).__init__(3 * nparticles - 4, 4 * nparticles)

    def map(self, xs):
        xs = interpret_array(xs, self.ndim_in)
        nparticles = self.nparticles
        e_cm = self.e_cm

        p = np.empty((xs.shape[0], nparticles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = np.zeros((xs.shape[0], nparticles))
        u = np.empty((xs.shape[0], nparticles - 2))

        Q = np.tile([e_cm, 0, 0, 0], (xs.shape[0], 1))
        Q_prev = np.empty((xs.shape[0], 4))
        M[:, 0] = e_cm

        for i in range(2, nparticles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != nparticles:
                u[:, i - 2] = [
                    brentq(lambda x: ((nparticles + 1 - i) *
                                      x ** (2 * (nparticles - i)) -
                                      (nparticles - i) *
                                      x ** (2 * (nparticles + 1 - i)) - r_i),
                           0., 1.)
                    for r_i in xs[:, i - 2]]
                M[:, i - 1] = np.product(u[:, :i - 1], axis=1) * e_cm

            cos_theta = 2 * xs[:, nparticles - 6 + 2 * i] - 1
            phi = 2 * np.pi * xs[:, nparticles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2],
                                                        M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * np.cos(phi) * np.sqrt(1 - cos_theta ** 2)
            p[:, i - 2, 2] = q * np.sin(phi) * np.sqrt(1 - cos_theta ** 2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = np.sqrt(q ** 2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2], MINKOWSKI)
            Q = boost(Q_prev, Q, MINKOWSKI)

        p[:, nparticles - 1] = Q

        return p.reshape((xs.shape[0], nparticles * 4))

    def map_inverse(self, p):
        count = p.size // (self.nparticles * 4)
        p = p.reshape((count, self.nparticles, 4))

        M = np.empty(p.shape[0])
        M_prev = np.empty(p.shape[0])
        Q = np.empty((p.shape[0], 4))
        r = np.empty((p.shape[0], 3*self.nparticles-4))

        Q[:] = p[:, -1]

        for i in range(self.nparticles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i-2:].sum(axis=1)
            M = np.sqrt(np.einsum('ij,jk,ik->i', P, MINKOWSKI, P))

            if i != self.nparticles:
                u = M_prev/M
                r[:, i-2] = (self.nparticles+1-i)*u**(2*(self.nparticles-i)) - (self.nparticles-i)*u**(2*(self.nparticles+1-i))

            Q += p[:, i-2]
            p_prime = boost(np.einsum('ij,ki->kj', MINKOWSKI, Q), p[:, i-2], MINKOWSKI)
            r[:, self.nparticles-6+2*i] = .5 * (p_prime[:, 3]/np.sqrt(np.sum(p_prime[:, 1:]**2, axis=1)) + 1)
            phi = np.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.nparticles-5+2*i] = phi/(2*np.pi) + (phi<0)

        return r

    def jac(self, xs):
        return np.ones(xs.shape[0])

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = ((np.pi / 2.) ** (nparticles - 1) * e_cm ** (2 * nparticles - 4) /
               (gamma(nparticles) * gamma(nparticles - 1)))
        return vol

    def pdf_gradient(self, xs):
        return 0
    
class RamboOnDietHadron(PhaseSpaceMapping):
    """ Rambo on diet for Hadron colliders with masses"""

    def __init__(self, e_had: float, nparticles: int, masses: list=None):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super(RamboOnDietHadron, self).__init__(3 * nparticles - 2, 4 * nparticles)
        
        self.e_had = e_had
        self.nparticles = nparticles
        self.masses = masses
        
        # Make sure the list is as long as the number of
        # finals state particles if given
        if self.masses:
            assert len(self.masses) == self.nparticles
        
        # Define min energy due to masses (no cuts etc)
        e_min = np.sum(self.masses) if self.masses else 0
        self.tau_min = (e_min/self.e_had)**2
 
    def _get_parton_fractions(self, r):
        
        if self.tau_min > 0:
            logtau = r[0] * np.log(self.tau_min)
        else:
            logtau = np.log(r[0])
        
        logx1 = (1 - r[1]) * logtau
        logx2 = r[1] * logtau 
        return np.exp(logx1), np.exp(logx2)
           
    def _get_pdf_random_numbers(self, x):
        tau = x[0] * x[1]
        r1 = np.log(tau)/np.log(self.tau_min)
        r2 = np.log(x[0])/np.log(tau)
        return r1, r2
        
    def _get_rapidity_and_fractions(self, q):
        tau = (np.einsum('ij,jk,ik->i', q[:,0,:], MINKOWSKI, q[:,0,:]) / self.e_had**2)[...,None]
        rapidity = np.arctanh(q[:,:,3]/q[:,:,0])
        logx1 = 0.5 * np.log(tau) + 0.5 * np.log((q[:,:,0] - q[:,:,3])/(q[:,:,0] + q[:,:,3]))
        logx2 = 0.5 * np.log(tau) - 0.5 * np.log((q[:,:,0] - q[:,:,3])/(q[:,:,0] + q[:,:,3]))
        return rapidity, np.exp(logx1), np.exp(logx2)
            

    def map(self, xs):
        xs = interpret_array(xs, self.ndim_in)
        xs, r1, r2 = xs[:,2:], xs[:, [0]], xs[:, [1]]
        
        # get partonic energies and boost variables
        x1, x2 = self._get_parton_fractions([r1, r2])
        rapidity = 0.5 * np.log(x1/x2)
        e_cm = self.e_had * np.sqrt(x1 * x2)
        
        p = np.empty((xs.shape[0], self.nparticles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = np.zeros((xs.shape[0], self.nparticles))
        u = np.empty((xs.shape[0], self.nparticles - 2))

        Q = e_cm * np.tile([1, 0, 0, 0], (xs.shape[0], 1))
        Q_prev = np.empty((xs.shape[0], 4))
        M[:, 0] = e_cm[:, 0]

        for i in range(2, self.nparticles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != self.nparticles:
                u[:, i - 2] = [
                    brentq(lambda x: ((self.nparticles + 1 - i) *
                                      x ** (2 * (self.nparticles - i)) -
                                      (self.nparticles - i) *
                                      x ** (2 * (self.nparticles + 1 - i)) - r_i),
                           0., 1.)
                    for r_i in xs[:, i - 2]]
                M[:, i - 1] = np.product(u[:, :i - 1], axis=1) * e_cm[:, 0]

            cos_theta = 2 * xs[:, self.nparticles - 6 + 2 * i] - 1
            phi = 2 * np.pi * xs[:, self.nparticles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2],
                                                        M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * np.cos(phi) * np.sqrt(1 - cos_theta ** 2)
            p[:, i - 2, 2] = q * np.sin(phi) * np.sqrt(1 - cos_theta ** 2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = np.sqrt(q ** 2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2], MINKOWSKI)
            Q = boost(Q_prev, Q, MINKOWSKI)

        p[:, self.nparticles - 1] = Q
        
        if self.masses:
            # Define masses
            m = np.tile(self.masses, (xs.shape[0], 1))
            
            # solve for massive case
            xi = np.empty((xs.shape[0], 1, 1))
            xi[:, 0, 0] = [brentq(lambda x: ( np.sum( np.sqrt(m[i, :]**2 + x**2 * p[i, :, 0]**2), axis=-1) - e_cm[i,0]), 0., 1.) for i in range(xs.shape[0])]
            
            # Make them massive
            k = np.empty((xs.shape[0], self.nparticles, 4))
            k[:, :, 0] = np.sqrt(m**2 + xi[:,:,0]**2 * p[:, :, 0]**2)
            k[:, :, 1:] = xi * p[:, :, 1:]
            
            # Boost into hadronic lab frame
            k = boost_z(k, rapidity, inverse=False)

            return k.reshape((xs.shape[0], self.nparticles * 4))
        
        # Boost into hadronic lab frame
        p = boost_z(p, rapidity, inverse=False)
        
        return p.reshape((xs.shape[0], self.nparticles * 4))

    def map_inverse(self, k):
        count = k.size // (self.nparticles * 4)
        k = k.reshape((count, self.nparticles, 4))

        M = np.empty(k.shape[0])
        M_prev = np.empty(k.shape[0])
        Q = np.empty((k.shape[0], 4))
        r = np.empty((k.shape[0], 3*self.nparticles-4))
        
        # Boost into partonic CM frame and get x1 and x2
        q = np.sum(k, axis=1, keepdims=True)
        rapidity, x1, x2 = self._get_rapidity_and_fractions(q)
        e_cm = self.e_had * np.sqrt(x1 * x2)
        k = boost_z(k, rapidity, inverse=True)
        

        # Make momenta massless
        p = np.empty((k.shape[0], self.nparticles, 4))
        if self.masses:
            # Define masses
            m = np.tile(self.masses, (k.shape[0], 1))
            
            # solve for mass case
            xi = np.empty((k.shape[0], 1, 1))
            xi[:, 0, 0] = [brentq(lambda x: ( np.sum( np.sqrt(k[i, :, 0]**2 - m[i, :]**2), axis=-1) - x * e_cm[i,0]), 0., 1.) for i in range(k.shape[0])]
            # Make them massive
            p[:, :, 0] = np.sqrt( k[:, :, 0]**2 - m**2)/xi[:,:,0]
            p[:, :, 1:] = k[:, :, 1:]/xi

        else:
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]
    

        Q[:] = p[:, -1]

        for i in range(self.nparticles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i-2:].sum(axis=1)
            M = np.sqrt(np.einsum('ij,jk,ik->i', P, MINKOWSKI, P))

            if i != self.nparticles:
                u = M_prev/M
                r[:, i-2] = (self.nparticles+1-i)*u**(2*(self.nparticles-i)) - (self.nparticles-i)*u**(2*(self.nparticles+1-i))

            Q += p[:, i-2]
            p_prime = boost(np.einsum('ij,ki->kj', MINKOWSKI, Q), p[:, i-2], MINKOWSKI)
            r[:, self.nparticles-6+2*i] = .5 * (p_prime[:, 3]/np.sqrt(np.sum(p_prime[:, 1:]**2, axis=1)) + 1)
            phi = np.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.nparticles-5+2*i] = phi/(2*np.pi) + (phi<0)

        # get additional random numbers for the pdfs
        r1, r2 = self._get_pdf_random_numbers([x1, x2])
        r = np.concatenate((r1, r2, r), axis=-1)
        
        return r

    def jac(self, xs):
        return np.ones(xs.shape[0])

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = ((np.pi / 2.) ** (nparticles - 1) * e_cm ** (2 * nparticles - 4) /
               (gamma(nparticles) * gamma(nparticles - 1)))
        return vol

    def pdf_gradient(self, xs):
        return 0
