import numpy as np
from scipy.optimize import brentq
from math import gamma

from .base import PhaseSpaceMapping
from .helper import interpret_array, map_fourvector_rambo, two_body_decay_factor, boost

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
