import torch
import torch.nn as nn

from survae.flows import Flow
from survae.distributions import StandardNormal, StandardUniform
from survae.transforms import (
    Reverse,
    Augment,
    Shuffle,
    ActNormBijection,
    AffineCouplingBijection,
    RationalQuadraticSplineCouplingBijection,
    CubicSplineCouplingBijection,
    Sigmoid,
    Logit
)
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, scale_fn

class BaseFlow(nn.Module):
    def __init__(
        self,
        in_dim: int,
        aug_dim: int,
        config,
        unit_hypercube: bool,
        steps_per_epoch: int,
        device=torch.device("cpu"),
    ):
        super().__init__() 
        self.in_dim = in_dim
        self.aug_dim = aug_dim
        assert aug_dim % 2 == 0
        self.n_units = config.n_units
        self.n_blocks = config.n_blocks
        self.n_layers = config.n_layers
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.config = config
        self.unit_hypercube  = unit_hypercube

    def define_model_architecture(self):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _forward(...) method"
        )

    def set_optimizer(self):

        self.optim = torch.optim.Adam(
            self.params_trainable,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=1e-6,
            weight_decay=self.config.weight_decay,
        )
        
        lr_decay = self.config.exp_decay
        lr_scheduler = self.config.lr_scheduler
        
        if lr_scheduler == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim, max_lr=self.config.max_lr, steps_per_epoch=self.steps_per_epoch, epochs=self.config.n_epochs
            )
        elif lr_scheduler == "exponential":
            weight_updates = self.steps_per_epoch * self.config.n_epochs
            decay_rate = lr_decay ** (1 / max(weight_updates, 1))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optim, gamma=decay_rate
            )
        else: 
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optim, step_size=1, gamma=self.config.gamma
            )
        
    def forward(self, z):
        return self.model.sample_refined(z)

    def save(self, name: str):
        torch.save(
            {"opt": self.optim.state_dict(), "net": self.model.state_dict()}, name
        )

    def load(self, name: str):
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError:
            print("Cannot load optimizer")
            

class AffineFlow(BaseFlow):
    def __init__(
        self,
        in_dim: int,
        aug_dim: int,
        config,
        unit_hypercube: bool,
        steps_per_epoch: int,
        device=torch.device("cpu"),
    ):
        super().__init__(in_dim, aug_dim, config, unit_hypercube, steps_per_epoch, device=device) 
        

    def define_model_architecture(self):
        
        A = self.in_dim + self.aug_dim
        SPLITS = [A - A // 2, A // 2]
        P = 2

        hidden_units = [self.n_units for _ in range(self.n_layers)]

        if self.unit_hypercube:
            transforms = [Logit()]
        else:
            transforms = []
        
        if self.aug_dim:
            print(f"Augmented Affine-Flow (dims = {self.in_dim} + {self.aug_dim})")
            transforms.append(Augment(StandardNormal((self.aug_dim,)), x_size=self.in_dim))
        else:
            print(f"Baseline Affine-Flow (dims = {self.in_dim})")

        for _ in range(self.n_blocks):
            net = nn.Sequential(
                MLP(SPLITS[0], P * SPLITS[1], hidden_units=hidden_units, activation='relu'), # was 'relu'
                ElementwiseParams(P),
            )
            transforms.append(
                AffineCouplingBijection(net, scale_fn=scale_fn("tanh_exp"))
            )
            transforms.append(ActNormBijection(A))
            transforms.append(Shuffle(A))
            #transforms.append(Reverse(A))
        transforms.pop()

        self.model = (
            Flow(base_dist=StandardNormal((A,)), transforms=transforms)
            .double()
            .to(self.device)
        )
        
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )


class RQSFlow(BaseFlow):
    def __init__(
        self,
        in_dim: int,
        aug_dim: int,
        config,
        unit_hypercube: bool,
        steps_per_epoch: int,
        device=torch.device("cpu"),
        n_bins: int = 10,
    ):
        super().__init__(in_dim, aug_dim, config, unit_hypercube, steps_per_epoch, device=device)
        self.n_bins = n_bins

    def define_model_architecture(self):
        A = self.in_dim + self.aug_dim
        SPLITS = [A - A // 2, A // 2]
        B = self.n_bins
        P = 3 * B + 1

        hidden_units = [self.n_units for _ in range(self.n_layers)]

        if not self.unit_hypercube:
            transforms = [Sigmoid()]
        else:
            transforms = []
            
        if self.aug_dim:
            print(f"Augmented RQS-Flow (dims = {self.in_dim} + {self.aug_dim})")
            transforms.append(Augment(StandardUniform((self.aug_dim,)), x_size=self.in_dim))
        else:
            print(f"Baseline RQS-Flow (dims = {self.in_dim})")
        
        for _ in range(self.n_blocks):
            net = nn.Sequential(
                MLP(SPLITS[0], P * SPLITS[1], hidden_units=hidden_units, activation="relu"),
                ElementwiseParams(P),
            )
            transforms.append(
                RationalQuadraticSplineCouplingBijection(net, num_bins=B)
            )
            transforms.append(Shuffle(A))
            #transforms.append(Reverse(A))
        transforms.pop()

        self.model = (
            Flow(base_dist=StandardUniform((A,)), transforms=transforms)
            .double()
            .to(self.device)
        )
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        
class RealRQSFlow(BaseFlow):
    def __init__(
        self,
        in_dim: int,
        aug_dim: int,
        config,
        unit_hypercube: bool,
        steps_per_epoch: int,
        device=torch.device("cpu"),
        n_bins: int = 10,
    ):
        super().__init__(in_dim, aug_dim, config, unit_hypercube, steps_per_epoch, device=device)
        self.n_bins = n_bins

    def define_model_architecture(self):
        A = self.in_dim + self.aug_dim
        SPLITS = [A - A // 2, A // 2]
        B = self.n_bins
        P = 3 * B + 1

        hidden_units = [self.n_units for _ in range(self.n_layers)]

        if not self.unit_hypercube:
            transforms = [Sigmoid()]
        else:
            transforms = []
            
        if self.aug_dim:
            print(f"Augmented RQS-Flow (dims = {self.in_dim} + {self.aug_dim})")
            transforms.append(Augment(StandardUniform((self.aug_dim,)), x_size=self.in_dim))
        else:
            print(f"Baseline RQS-Flow (dims = {self.in_dim})")
        
        for _ in range(self.n_blocks):
            net = nn.Sequential(
                MLP(SPLITS[0], P * SPLITS[1], hidden_units=hidden_units, activation="relu"),
                ElementwiseParams(P),
            )
            transforms.append(
                RationalQuadraticSplineCouplingBijection(net, num_bins=B)
            )
            transforms.append(Shuffle(A))
            #transforms.append(Reverse(A))
        transforms.pop()
        
        transforms.append(Logit())

        self.model = (
            Flow(base_dist=StandardNormal((A,)), transforms=transforms)
            .double()
            .to(self.device)
        )
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
            

class CubicFlow(BaseFlow):
    def __init__(
        self,
        in_dim: int,
        aug_dim: int,
        config,
        unit_hypercube: bool,
        steps_per_epoch: int,
        device=torch.device("cpu"),
        n_bins: int = 10,
    ):
        super().__init__(in_dim, aug_dim, config, unit_hypercube, steps_per_epoch, device=device)
        self.n_bins = n_bins

    def define_model_architecture(self):

        A = self.in_dim + self.aug_dim
        SPLITS = [A - A // 2, A // 2]
        B = self.n_bins # Heidelberg used 60
        P = 2 * B + 2

        hidden_units = [self.n_units for _ in range(self.n_layers)]

        if not self.unit_hypercube:
            transforms = [Sigmoid()]
        else:
            transforms = []
            
        if self.aug_dim:
            print(f"Augmented Cubic-Flow (dims = {self.in_dim} + {self.aug_dim})")
            transforms.append(Augment(StandardUniform((self.aug_dim,)), x_size=self.in_dim))
        else:
            print(f"Baseline Cubic-Flow (dims = {self.in_dim})")
        
        for _ in range(self.n_blocks):
            net = nn.Sequential(
                MLP(SPLITS[0], P * SPLITS[1], hidden_units=hidden_units, activation="relu"),
                ElementwiseParams(P),
            )
            transforms.append(CubicSplineCouplingBijection(net, num_bins=B))
            transforms.append(Shuffle(A))
            #transforms.append(Reverse(A))
        transforms.pop()

        self.model = (
            Flow(base_dist=StandardUniform((A,)), transforms=transforms)
            .double()
            .to(self.device)
        )
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )