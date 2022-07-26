
class PhaseSpaceMapping:
    def __init__(self, ndim_in, ndim_out):
        self.ndim_in = ndim_in  # input dimensionality
        self.ndim_out = ndim_out

    def pdf(self, xs):
        raise NotImplementedError

    def pdf_gradient(self, xs):
        raise NotImplementedError

    def map(self, xs):
        raise NotImplementedError

    def map_inverse(self, xs):
        raise NotImplementedError

