""" Define AugFeatures module """

from typing import Tuple, Iterable, Union, Callable, List
import torch
import torch.nn as nn
from .preprocess import Scaler


class _Augment(nn.Module):
    """Base classe that defines the augmentation"""

    def __init__(
        self,
        n_particles: int,
        Es: bool = 1,
        Px: bool = 1,
        Py: bool = 1,
        Pz: bool = 1,
        PT: bool = 0,
        Eta: bool = 0,
        Phi: bool = 0,
        logE: bool = 0,
        logPT: bool = 0,
        logPTcut: List[bool, Iterable[float]] = [0, None],
        logEtacut: List[bool, Iterable[float]] = [0, None],
        DeltaR: Iterable[Tuple[int]] = None,
        Mij: Iterable[Tuple[int]] = None,
        DeltaPhi: Iterable[Tuple[int]] = None,
        DeltaEta: Iterable[Tuple[int]] = None,
        **kwargs,
    ):

        super(_Augment, self).__init__(**kwargs)

        self.epsilon = 1e-10
        self.n_particles = n_particles

        self.Es = Es
        self.Px = Px
        self.Py = Py
        self.Pz = Pz

        self.PT = PT
        self.Eta = Eta
        self.Phi = Phi

        self.logE = logE
        self.logPT = logPT
        self.logPTcut = logPTcut
        self.logEtacut = logEtacut

        self.DeltaR = DeltaR
        self.Mij = Mij
        self.DeltaPhi = DeltaPhi
        self.DeltaEta = DeltaEta

        self.n_out = (
            int(self.Es)
            + int(self.Px)
            + int(self.Py)
            + int(self.Pz)
            + int(self.PT)
            + int(self.Eta)
            + int(self.Phi)
            + int(self.logE)
            + int(self.logPT)
            + int(self.logPTcut[0])
            + int(self.logEtacut[0])
        )

        self.n_aug = (
            len(self.DeltaR) + len(self.Mij) + len(self.DeltaPhi) + len(self.DeltaEta)
        )

    def _reparametrize(self, x):

        out = []
        aug = []

        # x.shape:     (b, 4 * n_particles)
        # reshape   -> (b, n_particles, 4)
        # transpose -> (b, 4, n_particles)
        x = torch.reshape(x, (x.shape()[0], self.n_particles, 4)).transpose(1, 2)

        # get E, Px, Py, Pz
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]

        # Get absolute 3-momentum |p|
        Ps = torch.sqrt(torch.clamp(Xs ** 2 + Ys ** 2 + Zs ** 2, min=self.epsilon))

        if self.Es:
            out.append(Es)
        if self.Px:
            out.append(Xs)
        if self.Py:
            out.append(Ys)
        if self.Pz:
            out.append(Zs)

        if self.PT:
            PTs = torch.sqrt(torch.clamp(Xs ** 2 + Ys ** 2, min=self.epsilon))
            out.append(PTs)

        if self.Eta:
            etas = 0.5 * (
                torch.log(torch.clamp(Ps + Zs, min=self.epsilon))
                - torch.log(torch.clamp(Ps - Zs, min=self.epsilon))
            )
            out.append(etas)

        if self.Phi:
            phis = torch.atan2(Ys, Xs)
            out.append(phis)

        if self.logE:
            out.append(torch.log(Es))

        if self.logPT:
            PTs = torch.sqrt(torch.clamp(Xs ** 2 + Ys ** 2, min=self.epsilon))
            out.append(torch.log(PTs))

        if self.logPTcut != None:
            PTs = torch.sqrt(torch.clamp(Xs ** 2 + Ys ** 2, min=self.epsilon))
            cuts = torch.tensor(self.logPTcut[1]).double()
            logptcut = torch.log(PTs - cuts)
            out.append(logptcut)

        if self.logEtacut != None:
            etas = 0.5 * (
                torch.log(torch.clamp(Ps + Zs, min=self.epsilon))
                - torch.log(torch.clamp(Ps - Zs, min=self.epsilon))
            )
            cuts = torch.tensor(self.logetacut[1]).double()
            logetacut = torch.log((cuts + etas) / (cuts - etas))
            out.append(logetacut)

        # additional augmented features
        if self.DeltaR:

            Etas = 0.5 * (
                torch.log(torch.clamp(Ps + Zs, min=self.epsilon))
                - torch.log(torch.clamp(Ps - Zs, min=self.epsilon))
            )
            Phis = torch.atan2(Ys, Xs)

            for id1, id2 in self.DeltaR:

                DeltaEta = torch.abs(Etas[:, id1] - Etas[:, id2])

                Deltaphi = torch.abs(Phis[:, id1] - Phis[:, id2])
                Deltaphi = torch.where(
                    Deltaphi > torch.pi, 2.0 * torch.pi - Deltaphi, Deltaphi
                )

                DeltaR = torch.sqrt(
                    torch.clamp(DeltaEta ** 2 + Deltaphi ** 2, self.epsilon)
                )[..., None]

                aug.append(DeltaR)

        if self.Mij:

            for ids in self.Mij:
                Ei = 0.0
                pxi = 0.0
                pyi = 0.0
                pzi = 0.0

                for idx in ids:
                    Ei += Es[:, idx]
                    pxi += Xs[:, idx]
                    pyi += Ys[:, idx]
                    pzi += Zs[:, idx]

                Mij2 = Ei ** 2 - pxi ** 2 - pyi ** 2 - pzi ** 2
                Mij = torch.sqrt(torch.clamp(Mij2, min=self.epsilon))[..., None]

                aug.append(Mij)

        if self.DeltaPhi:

            Phis = torch.atan2(Ys, Xs)

            for id1, id2 in self.DeltaPhi:

                Deltaphi = torch.abs(Phis[:, id1] - Phis[:, id2])
                Deltaphi = torch.where(
                    Deltaphi > torch.pi, 2.0 * torch.pi - Deltaphi, Deltaphi
                )[..., None]

                aug.append(Deltaphi)

        if self.DeltaEta:

            Etas = 0.5 * (
                torch.log(torch.clamp(Ps + Zs, min=self.epsilon))
                - torch.log(torch.clamp(Ps - Zs, min=self.epsilon))
            )

            for id1, id2 in self.DeltaEta:

                DeltaEta = torch.abs(Etas[:, id1] - Etas[:, id2])[..., None]

                aug.append(DeltaEta)

        # y.shape:     (b, n_out, n_particles)
        # transpose -> (b, n_particles, n_out)
        # reshape   -> (b, n_out * n_particles)
        shape_dim = self.n_out * self.n_particles
        x_out = torch.stack(out, dim=1)
        x_out = torch.transpose(x_out, 1, 2).reshape(x_out.shape[0], shape_dim)

        if self.n_aug > 0:
            x_aug = torch.cat(aug, dim=-1)
            return torch.cat([x_out, x_aug], dim=-1)

        return x_out

    def forward(self, x):
        """Needs to be defined by the subclasses"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide forward(...) method"
        )


class AugFeatures(_Augment):
    """A module for agumenting additional
    features to the 4-vector prescription
    """

    def __init__(self, scaler_fn: Union[float, Scaler], *args, **kwargs):
        """See base class docstring for all args and kwargs"""
        super(AugFeatures, self).__init__(*args, **kwargs)

        self.scaler_fn = scaler_fn

    def _preprocess(self, x, inverse=False):

        if not inverse:
            if isinstance(self.scaler_fn, Scaler):
                return self.scaler_fn.transform(x)
            elif isinstance(self.scaler_fn, float):
                return x / self.scaler_fn
            else:
                raise ValueError("Scaler function must be either a callable or a float")
        else:
            if isinstance(self.scaler_fn, Scaler):
                return self.scaler_fn.inverse_transform(x)
            elif isinstance(self.scaler_fn, float):
                return x * self.scaler_fn
            else:
                raise ValueError("Scaler function must be either a callable or a float")

    def forward(self, x):

        # undo preprocessing of data
        x = self._preprocess(x, inverse=True)

        # do the reparametrization
        x = self._reparametrize(x)

        return x
