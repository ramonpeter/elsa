""" Define MCMC modules """

import torch
from torch.autograd import Variable, grad
import torch.nn as nn

class HamiltonMCMC(nn.Module):
	"""
 	Hamilton Markov-Chain Monte Carlo
	which samples from an energy-based model
	based on nn.Module.
	"""

	def __init__(
		self,
		generator: nn.Module,
		classifier: nn.Module,
		latent_dim: int,
		M: torch.Tensor = None,
		L: int=100,
		eps: float=1e-2,
		n_chains: int=1,
		burnin: int=1000,
		device=torch.device("cpu"),
	):
		"""
		Args:
			generator (nn.Module): Module to generate.
			classifier (nn.Module): Energy-based module.
			latent_dim (int): Dimension of the latent space.
			M (torch.Tensor, optional): Typical correlation Matrix. Defaults to None.
			L (int, optional): Leapfrog steps. Defaults to 100.
			eps (float, optional): Step size. Defaults to 1e-2.
			n_chains (int, optional): Parallelize the sampling. Defaults to 1.
			burnin (int, optional): Number of events not used in final sample. Defaults to 1000.
			device (torch.devie, optional): the chosen device to be used.
		"""
		super(HamiltonMCMC, self).__init__()

		self.generator = generator
		self.classifier = classifier
		self.latent_dim = latent_dim
		self.burnin = burnin
		self.device = device

		if M == None:
			self.M = torch.diag(torch.Tensor([1] * self.latent_dim))
		else:
			self.M = M

		self.L = L
		self.eps = eps
		self.n_chains = n_chains

	def U(self, q):
		sq_norm = torch.sum(torch.square(q), dim=-1, keepdim=True)

		return sq_norm / 2 - self.classifier(self.generator(q)) # define forward pass as sample_refined

	def grad_U(self, q):

		q.requires_grad = True
		U_grad = grad(self.U(q).sum(), q)[0]
		q = q.detach()

		return U_grad

	def leapfrog_step(self, q_init):

		q = q_init
		p_init = torch.randn(q.shape).detach().to(self.device)
		p = p_init.detach()

		# Make half a step for momentum at the beginning
		p = p - self.eps * self.grad_U(q) / 2

		q = q.detach()
		q_init = q_init.detach()

		# Alternate full steps for position and momentum
		for i in range(self.L):
			# full step position
			with torch.no_grad():
				q = q + self.eps * p
			# make full step momentum, except at end of trajectory
			if i != self.L - 1:
				p = p - self.eps * self.grad_U(q)

		# Make half step for momentum at the end
		p = p - self.eps * self.grad_U(q) / 2
		# Negate momentum at and of trajectory to make proposal symmetric
		p = p * -1

		q = q.detach()

		# Evaluate potential and kinetic energies
		with torch.no_grad():
			U_init = self.U(q_init)
			K_init = torch.sum(torch.square(p_init), dim=-1, keepdim=True) / 2
			U_proposed = self.U(q)
			K_proposed = torch.sum(torch.square(p), dim=-1, keepdim=True) / 2

		u = torch.rand(self.n_chains, 1).to(self.device)
		mask = (u < torch.exp(U_init - U_proposed + K_init - K_proposed)).flatten()

		q[~mask] = q_init[~mask]

		return q, torch.sum(mask).cpu().detach().numpy()

	def sample(self, latent_dim, n_samples):
		q = (
			torch.randn((self.n_chains, latent_dim))
			.double()
			.detach()
			.to(self.device)
		)
		sample = []
		accepted = 0

		# Burn in
		for j in range(self.burnin):
			q, _ = self.leapfrog_step(q)
			if j % 1000 == 0:
				print(j)
		print("end burn in")

		# actual sampling
		for i in range(n_samples):
			q, acc = self.leapfrog_step(q)
			accepted += acc
			sample.append(q)
			if i % 100 == 0:
				print(accepted)

		acc_rate = accepted / (self.n_chains * n_samples)
		return torch.cat(sample), acc_rate
