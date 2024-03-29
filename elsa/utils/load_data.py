""" Data loader functions"""

import os
import torch
import numpy as np
import pandas as pd
from ..modules.preprocess import (
    Scaler,
    SimpleScaler,
    FourMomScaler,
    ThreeMomPlusScaler,
    ThreeMomScaler,
    MinimalRepScaler,
    PrecisionEnthusiastScaler,
    EnlargedFeaturSpaceScaler,
    MahamboScaler,
    AugmentedMahamboFeatureScaler,
    
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
			self.disc_scaler = self.get_scaler(self.data, conf.disc_scaler, is_disc=True)
		else:
			self.disc_scaler = self.gen_scaler
		self.is_hypercube = self.gen_scaler.is_hypercube

		# get dataset
		self.train_data, self.test_data, self.shape = self.prepare_dataset(
			self.data, self.gen_scaler, True
		)
  
		# get dataset with other scaler
		if conf.disc_scaler is not None:
			self.disc_train_data, self.disc_test_data, self.disc_shape = self.prepare_dataset(
				self.data, self.disc_scaler, False
			)
		else:
			self.disc_train_data = self.train_data
			self.disc_test_data = self.test_data
			self.disc_shape = self.shape

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

	def get_scaler(self, data: np.ndarray, scl_str: str, is_disc: bool=False):
		if scl_str == "Simple":
			if self.conf.scale is not None:
				scales = self.conf.scale
				means = 1.0
			else:
				scales = np.std(data, 0)
				means = np.mean(data, 0)
			scaler = SimpleScaler(scales, means)
		elif scl_str == "FourMom":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = FourMomScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "ThreeMomPlus":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = ThreeMomPlusScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "ThreeMom":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = ThreeMomScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "MinRep":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = MinimalRepScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "Precisesiast":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = PrecisionEnthusiastScaler(
				self.e_had, self.nparticles, self.masses, ptcuts=self.ptcuts, is_disc_scaler=is_disc
			)
		elif scl_str == "Elfs":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = EnlargedFeaturSpaceScaler(
				self.e_had, self.nparticles, self.masses, ptcuts=self.ptcuts
			)
		elif scl_str == "Mahambo":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = MahamboScaler(self.e_had, self.nparticles, self.masses)
		elif scl_str == "Amber":
			assert self.toy == False, "`{scl_str}`-scaler cannot be used with toy example"
			scaler = AugmentedMahamboFeatureScaler(self.e_had, self.nparticles, self.masses)
		else:
			raise ValueError("Scaler is not implemented")

		return scaler

	def prepare_dataset(self, data: np.ndarray, scaler: Scaler, flow: bool=True):
		# preprocess events
		events = scaler.fit_and_transform(data)
		# split into train and validate
		events_train = events[: self.split]
		events_validate = events[self.validate_split :]

		shape = events_train.shape[1]
		model = "flow" if flow else "disc"
		print(f"{model} data shape: {events_train.shape}")

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
