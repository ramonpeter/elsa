import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform


class Flow(Distribution):
	"""
	Base class for Flow.
	Flows use the forward transforms to transform data to noise.
	The inverse transforms can subsequently be used for sampling.
	These are typically useful as generative models of data.
	"""

	def __init__(self, base_dist, transforms):
		super(Flow, self).__init__()
		assert isinstance(base_dist, Distribution)
		if isinstance(transforms, Transform): transforms = [transforms]
		assert isinstance(transforms, Iterable)
		assert all(isinstance(transform, Transform) for transform in transforms)
		self.base_dist = base_dist
		self.transforms = nn.ModuleList(transforms)
		self.lower_bound = any(transform.lower_bound for transform in transforms)

	def log_prob(self, x):
		log_prob = torch.zeros(x.shape[0], device=x.device)
		for transform in self.transforms:
			x, ldj = transform(x)
			log_prob += ldj
		log_prob += self.base_dist.log_prob(x)
		return log_prob

	def sample(self, num_samples):
		# Modified to also return the latent
		z = self.base_dist.sample(num_samples)
		latent = z
		for transform in reversed(self.transforms):
			z = transform.inverse(z)
		return z, latent

	def sample_refined(self, z):
		# sample from refine latent space
		for transform in reversed(self.transforms):
			z = transform.inverse(z)
		return z

	def encode(self, x):
		# Added to also just give the forward pass
		for transform in self.transforms:
			x, _ = transform(x)
		return x

	def sample_with_log_prob(self, num_samples):
		raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
