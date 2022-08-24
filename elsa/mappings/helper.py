import numpy as np

def interpret_array(array, ndim=None):
    array = np.array(array, copy=False, subok=True, ndmin=1)
    if array.ndim == 2:
        if ndim and ndim != array.shape[1]:
            raise RuntimeWarning("Unexpected dimension of entries in array.")
        return array

    if ndim is None:
        # can't distinguish 1D vs ND cases. Treat as 1D case
        return array[:, np.newaxis]

    if array.ndim == 1:
        if ndim == 1:
            # many 1D entries
            return array[:, np.newaxis]
        elif array.size == ndim:
            # one ndim-dim entry
            return array[np.newaxis, :]
        else:
            raise RuntimeError("Bad array shape.")
        
def map_fourvector_rambo(xs):
    """ Transform unit hypercube points into into four-vectors. """
    c = 2. * xs[:, :, 0] - 1.
    phi = 2. * np.pi * xs[:, :, 1]

    q = np.empty_like(xs)
    q[:, :, 0] = -np.log(xs[:, :, 2] * xs[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.cos(phi)
    q[:, :, 2] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.sin(phi)
    q[:, :, 3] = q[:, :, 0] * c

    return q


def two_body_decay_factor(M_i_minus_1, M_i, m_i_minus_1):
    return 1./(8*M_i_minus_1**2) * np.sqrt((M_i_minus_1**2 - (M_i+m_i_minus_1)**2)*(M_i_minus_1**2 - (M_i-m_i_minus_1)**2))


def boost(q, ph, metric):
    p = np.empty(q.shape)

    rsq = np.sqrt(np.einsum('kd,dd,kd->k', q, metric, q))

    p[:, 0] = np.einsum('ki,ki->k', q, ph) / rsq
    c1 = (ph[:, 0]+p[:, 0]) / (rsq+q[:, 0])
    p[:, 1:] = ph[:, 1:] + c1[:, np.newaxis]*q[:, 1:]

    return p

def boost_z(q, rapidity, inverse=False):
    p = np.empty(q.shape)
    sign = -1.0 if inverse else 1.0
    p[:, :, 0] = q[:, :, 0] * np.cosh(rapidity) + sign * q[:, :, 3] * np.sinh(rapidity)
    p[:, :, 1] = q[:, :, 1]
    p[:, :, 2] = q[:, :, 2]
    p[:, :, 3] = q[:, :, 3] * np.cosh(rapidity) + sign * q[:, :, 0] * np.sinh(rapidity)

    return p