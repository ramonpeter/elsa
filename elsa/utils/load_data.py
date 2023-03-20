""" Data loader functions"""

import os
import torch
import numpy as np
import pandas as pd
from ..modules.preprocess import (
    Scaler,
    RamboScaler,
    SimpleScaler,
    SchumannScaler,
    ThreeMomScaler,
    MinimalRepScaler,
    HeimelScaler,
)


class Loader:
	def __init__(self, conf, device):
		self.conf = conf
		self.device = device

		# Define process dependent data
		self.e_had = conf.e_had
		self.ptcuts = conf.ptcuts
		self.masses = conf.masses

		# initialize data
		self.data = self.read_files(conf.datapath, conf.dataset)
		self.define_splitting(self.data)

		# Make sure number of particles is consistent
		if conf.datapath == "../../datasets/lhc/":
			self.nparticles = self.data.shape[1] // 4
			assert (
				len(self.masses) == self.nparticles
			), "Number of masses must be equal to particles"
			assert (
				len(self.ptcuts) == self.nparticles
			), "Number of ptcuts must be equal to particles"
			self.toy = False
		else:
			self.nparticles = None
			self.toy = True

		# define scaler for gen and discrimination
		self.gen_scaler = self.get_scaler(self.data, conf.gen_scaler)
		if conf.disc_scaler is not None:
			self.disc_scaler = self.get_scaler(self.data, conf.disc_scaler)
		else:
			self.disc_scaler = self.gen_scaler
		self.is_hypercube = self.gen_scaler.is_hypercube

		# get dataset
		self.train_data, self.test_data, self.shape = self.prepare_dataset(
			self.data, self.gen_scaler
		)

	def read_files(self, datapath, dataset, verbose=True):
		events = []
		for file in os.listdir(datapath):
			if dataset + ".h5" == file:
				if verbose:
					print("Reading data from: {}".format(file))
					events = pd.read_hdf(os.path.join(datapath, file)).values

		return events

	def define_splitting(self, data: np.ndarray):
		if self.conf.test == True:
			split = int(len(data) * 0.01)
		else:
			split = int(len(data) * 0.8)

		self.validate_split = int(len(data) * 0.95)
		self.split = split

	def get_scaler(self, data: np.ndarray, scl_str: str):
		if scl_str == "Simple":
			if self.conf.scale is not None:
				scales = self.conf.scale
				means = 1.0
			else:
				scales = np.std(data, 0)
				means = np.mean(data, 0)
			scaler = SimpleScaler(scales, means)
		elif scl_str == "Schumann":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = SchumannScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "Momenta":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = ThreeMomScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "Minrep":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = MinimalRepScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "Heimel":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = HeimelScaler(
				self.e_had, self.nparticles, self.masses, ptcuts=self.ptcuts
			)
		elif scl_str == "Rambo":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = RamboScaler(self.e_had, self.nparticles, self.masses)
		else:
			raise ValueError("Scaler is not implemented")

		return scaler

	def prepare_dataset(self, data: np.ndarray, scaler: Scaler):
		# preprocess events
		events = scaler.fit_and_transform(data)
		# split into train and validate
		events_train = events[: self.split]
		events_validate = events[self.validate_split :]

		shape = events_train.shape[1]
		print(f"data shape: {events_train.shape}")

		# Prepare train and validate data loaders
		train_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_train).to(self.device),
			batch_size=self.conf.batch_size,
			shuffle=True,
			drop_last=True,
		)

		validate_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_validate).to(self.device),
			batch_size=self.conf.batch_size,
			shuffle=False,
			drop_last=True,
		)

		return train_loader, validate_loader, shape
