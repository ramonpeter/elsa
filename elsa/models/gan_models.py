import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------


class netG(nn.Module):
    def __init__(
        self,
        in_dim=2,
        n_units=16,
        num_layers=1,
        latent_dim_gen=1,
        device=torch.device("cpu"),
        config=None,
    ):
        super(netG, self).__init__()

        self.in_dim = in_dim
        self.n_units = n_units
        self.num_layers = num_layers
        self.latent_dim_gen = latent_dim_gen
        self.device = device
        self.config = config

        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.parameters())
        )

    def define_model_architecture(self):

        model = nn.ModuleList()

        model.append(nn.Linear(self.latent_dim_gen * self.in_dim, self.n_units))
        # model.append(nn.ReLU())
        model.append(nn.LeakyReLU(0.1))

        for layer in range(self.num_layers):
            model.append(nn.Linear(self.n_units, self.n_units))
            # model.append(nn.ReLU())
            model.append(nn.LeakyReLU(0.1))

        model.append(nn.Linear(self.n_units, self.in_dim))

        self.model = model.double().to(self.device)
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def forward(self, x):
        for l in self.model:
            x = l(x)
        return x

    def set_optimizer(self):
        """
        Set optimizer for training
        """
        self.optim = torch.optim.Adam(
            self.params_trainable,
            lr=self.config.lr_ref,
            betas=self.config.betas,
            eps=1e-6,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optim, step_size=1, gamma=self.config.gamma
        )

    def save(self, name):
        torch.save({"opt": self.optim.state_dict(), "net": self.model.state_dict()}, name)
        
    def load(self, name):
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError:
            print("Cannot load optimizer for some reason or other")


# ----------------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------------


class netD(nn.Module):
    def __init__(
        self,
        in_dim=2,
        n_units=16,
        num_layers=1,
        steps_per_epoch = None,
        device=torch.device("cpu"),
        config=None,
    ):
        super(netD, self).__init__()

        self.in_dim = in_dim
        self.n_units = n_units
        self.num_layers = num_layers
        self.device = device
        self.config = config
        self.steps_per_epoch = steps_per_epoch

        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.parameters())
        )

    def define_model_architecture(self):

        model = nn.ModuleList()

        model.append(
            spectral_norm(nn.Linear(self.in_dim, self.n_units), n_power_iterations=2)
        )
        # model.append(nn.ReLU())
        model.append(nn.LeakyReLU(0.1))

        for layer in range(self.num_layers):
            model.append(
                spectral_norm(
                    nn.Linear(self.n_units, self.n_units), n_power_iterations=2
                )
            )
            # model.append(nn.ReLU())
            model.append(nn.LeakyReLU(0.1))

        model.append(spectral_norm(nn.Linear(self.n_units, 1), n_power_iterations=2))

        self.model = model.double().to(self.device)
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def define_model_architecture_unreg(self):

        model = nn.ModuleList()

        model.append(nn.Linear(self.in_dim, self.n_units))
        # model.append(nn.ReLU())
        model.append(nn.LeakyReLU(0.1))

        for layer in range(self.num_layers):
            model.append(nn.Linear(self.n_units, self.n_units))
            # model.append(nn.ReLU())
            model.append(nn.LeakyReLU(0.1))

        model.append(nn.Linear(self.n_units, 1))

        self.model = model.double().to(self.device)
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def forward(self, x):
        for l in self.model:
            x = l(x)
        return x

    def set_optimizer(self):
        """
        Set optimizer for training
        """
        self.optim = torch.optim.Adam(
            self.params_trainable,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=1e-6,
        )
        
        if self.steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim, max_lr=self.config.max_lr, steps_per_epoch=self.steps_per_epoch, epochs=self.config.n_epochs
            )
        else: 
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optim, step_size=1, gamma=self.config.gamma
            )

    def save(self, name):
        torch.save({"opt": self.optim.state_dict(), "net": self.model.state_dict()}, name)
        
    def load(self, name):
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        try:
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError:
            print("Cannot load optimizer for some reason or other")
