import torch
import torch.nn as nn

from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms import (
    Reverse,
    Augment,
    ActNormBijection,
    AffineCouplingBijection,
)
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, scale_fn


class INN(nn.Module):
    def __init__(
        self,
        in_dim=2,
        aug_dim=0, # if this in non-zero we have an augmented flow
        elwise_params=2,
        n_blocks=1,
        n_units=16,
        n_layers=1,
        actnorm = False,
        device=torch.device("cpu"),
        config=None,
    ):
        super(INN, self).__init__() 
        self.in_dim = in_dim
        self.aug_dim = aug_dim
        self.actnorm = actnorm
        assert aug_dim % 2 == 0
        self.elwise_params = elwise_params
        self.n_units = n_units
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.device = device
        self.config = config

    def define_model_architecture(self):

        A = self.in_dim + self.aug_dim
        P = self.elwise_params

        hidden_units = [self.n_units for _ in range(self.n_layers)]

        if self.aug_dim:
            print("Augmented Flow")
            transforms = [Augment(StandardNormal((self.aug_dim,)), x_size=self.in_dim)]
        else:
            print("Baseline Flow")
            transforms = []
        
        for _ in range(self.n_blocks):
            net = nn.Sequential(
                MLP(A // 2, P * A // 2, hidden_units=hidden_units, activation="relu"),
                ElementwiseParams(self.elwise_params),
            )
            transforms.append(
                AffineCouplingBijection(net, scale_fn=scale_fn("tanh_exp"))
            )
            if self.actnorm:
                transforms.append(ActNormBijection(A))
            transforms.append(Reverse(A))
        transforms.pop()

        self.model = (
            Flow(base_dist=StandardNormal((A,)), transforms=transforms)
            .double()
            .to(self.device)
        )
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def set_optimizer(self):

        self.optim = torch.optim.Adam(
            self.params_trainable,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=1e-6,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optim, step_size=1, gamma=self.config.gamma
        )

    def forward(self, z):
        return self.model.sample_refined(z)

    def save(self, name):
        torch.save(
            {"opt": self.optim.state_dict(), "net": self.model.state_dict()}, name
        )

    def load(self, name):
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError:
            print("Cannot load optimizer for some reason or other")