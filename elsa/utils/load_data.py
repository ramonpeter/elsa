""" Data loader functions"""

import os
import torch
import numpy as np
import pandas as pd
from ..modules.preprocess import PhysicsScaler, SimpleScaler


def read_files(DATAPATH, dataset, verbose=True):
	events = []
	for file in os.listdir(DATAPATH):
		if dataset + '.h5' == file:
			if verbose:
				print("Reading data from: {}".format(file))
				events = pd.read_hdf(os.path.join(DATAPATH, file)).values

	return events

def Loader(datapath: str, dataset: str, batch_size: int, test: bool, scale: float, weighted: bool, device):


	data = read_files(datapath, dataset)	

	if test == True:
		split = int(len(data) * 0.01)
	else:	
		split = int(len(data) * 0.8)

	validate_split = int(len(data) * 0.95)
	
	# Select a single global scale or one for each direction
	if weighted:
		if scale is not None:
			scales = scale
		else:
			scales = np.std(data[:,:-1],0)
		scaler = SimpleScaler(scales)
	elif datapath == '../../datasets/lhc/':
		e_had = 14000
		nparticles = data.shape[1] // 4
		masses = [80.419] + [0.] * (nparticles - 1)
		scaler = PhysicsScaler(e_had, nparticles, masses)
	else:
		if scale is not None:
			scales = scale
		else:
			scales = np.std(data,0)
		scaler = SimpleScaler(scales)
  
	# preprocess events
	events = scaler.transform(data)

	# split into train and validate
	events_train = events[:split]
	events_validate = events[validate_split:]

	shape = events_train.shape[1]
	print(f"data shape: {events_train.shape}")

	# Prepare train and validate data loaders
	train_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_train).to(device),
			batch_size = batch_size,
			shuffle = True,
			drop_last = True,
			)
	
	validate_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_validate).to(device),
			batch_size = batch_size,
			shuffle = False,
			drop_last = True,
			)

	return train_loader, validate_loader, split, shape, scaler
