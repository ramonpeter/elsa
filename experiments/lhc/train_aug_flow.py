""" Train augmented flow """

# Basics
import sys, os
import torch
import numpy as np
import pandas as pd

# Model
from elsa.models.flow_model import INN, CubicSplineFlow, RQSFlow

# Train utils
from elsa.utils.train_utils import AverageMeter, print_log, get_real_data, save_checkpoint
from elsa.utils.load_data import Loader

# Plotting
from elsa.utils.distributions import Distribution

# Load config and opts
import config_aug as c
import opts

###########
## Setup ##
###########

opts.parse(sys.argv, c)

config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
    if v[0]=='_': continue
    s=eval('c.%s'%(v))
    config_str += " {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(f"device: {device}")

###############
## Load data ##
###############

# TODO: do better preprocessing!
train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.datapath, c.dataset, c.batch_size, c.test, c.scaler, c.weighted, device)
scales_tensor = torch.tensor(scales).double().to(device)

if c.weighted:
	data_shape -= 1

print("\n" + "==="*30 + "\n")

##################
## Define Model ##
##################

flow = INN(in_dim=data_shape, aug_dim=c.aug_dim, n_blocks=c.n_blocks, n_units=c.n_units, n_layers=c.n_layers, device=device, config=c)
#flow = RQSFlow(in_dim=data_shape, aug_dim=c.aug_dim, n_blocks=c.n_blocks, n_units=c.n_units, n_layers=c.n_layers, device=device, config=c)
#flow = CubicSplineFlow(in_dim=data_shape, aug_dim=c.aug_dim, n_blocks=c.n_blocks, n_units=c.n_units, n_layers=c.n_layers, device=device, config=c)
flow.define_model_architecture() # This seems to be a bit annoying to call again?!
flow.set_optimizer()

print("\n" + "==="*30 + "\n")
#print(flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "==="*30 + "\n")

##############
## Training ##
##############

try:
	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/' + c.dataset + '/' + 'augflow/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/' +  c.dataset + '/' + 'augflow/n_epochs_' + str(c.n_epochs))

	F_loss_meter = AverageMeter()

	print('Training...')
	for epoch in range(c.n_epochs):
		for iteration in range(c.n_its_per_epoch_gen):

			i=0

			for data in train_loader:

				flow.model.train()
				flow.optim.zero_grad()

				events = data / scales_tensor

				f_loss = -flow.model.log_prob(events.to(device)).mean()

				F_loss_meter.update(f_loss.item())

				f_loss.backward()
				flow.optim.step()

				i += 1

			if epoch == 0 or (epoch + 1) % c.show_interval == 0:
				print_log(epoch+1, c.n_epochs, i, len(train_loader), flow.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, F_loss_meter)

			F_loss_meter.reset()

		if (epoch + 1) % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_F = {
					'epoch': epoch + 1,
					'model': flow.model.state_dict(),
					'optimizer': flow.optim.state_dict(),
					}
				save_checkpoint(checkpoint_F, log_dir + '/' + c.dataset + '/augflow/n_epochs_' + str(c.n_epochs), 'checkpoint_F_epoch_%03d' % (epoch+1))

			if c.test == True:
				size = 10000
			else:
				size = 300000

			with torch.no_grad():
				real = get_real_data(c.datapath, c.dataset, c.test, size)

				fake, _ = flow.model.sample(size)
				fake = fake.cpu().detach().numpy() * scales

			distributions = Distribution(real, fake, 'epoch_%03d' % (epoch+1) + '_target', 'AugFlow', log_dir + '/' + c.dataset + '/augflow/n_epochs_' + str(c.n_epochs), c.dataset)
			distributions.plot()
		
		flow.scheduler.step()
except:
	if c.checkpoint_on_error:
		flow.model.save(c.filename + '_ABORT')
	raise 


##############
## Sampling ##
##############

size = 1000000 # Should be part of the config

# Get flow samples
fake, _ = flow.model.sample(size)
fake = fake.cpu().detach().numpy()
fake *= scales

# Get real samples
real = get_real_data(c.datapath, c.dataset, c.test, size)

# Save to hdf5
s1 = pd.HDFStore(log_dir + '/' + c.dataset + '/augflow/n_epochs_' + str(c.n_epochs) + '/augflow.h5')
s1.append('data', pd.DataFrame(fake))
s1.close()

# Make plots
distributions = Distribution(real, fake, 'AugFlow', 'AugFlow', log_dir + '/' + c.dataset + '/augflow/n_epochs_' + str(c.n_epochs), c.dataset)
distributions.plot()
