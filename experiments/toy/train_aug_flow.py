""" Train augmented flow """
import sys, os
import torch

# Model
from augflow_model import AugFlow

# Data
from elsa.load_data import Loader

# Train utils
from elsa.utils.train_utils import AverageMeter, print_log, get_real_data, save_checkpoint

# Plotting
from elsa.utils.plotting.distributions import *
from elsa.utils.plotting.plots import *

###########
## Setup ##
###########

import config_LSR as c
import opts
opts.parse(sys.argv)

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

sys.exit()

###############
## Load data ##
###############

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.weighted)
scales_tensor = torch.Tensor(scales).double().to(device)

if c.weighted:
	data_shape -= 1

print("\n" + "==="*30 + "\n")

##################
## Define Model ##
##################

flow = AugFlow(in_dim=data_shape, aug_dim=c.aug_dim, n_blocks=c.n_blocks, internal_size=c.n_units, n_layers=c.n_layers, init_zeros=False, dropout=False)
flow.define_model_architecture() # This seems to be a bit annoying to call again?!
flow.set_optimizer()

print("\n" + "==="*30 + "\n")
print(flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "==="*30 + "\n")


##############
## Training ##
##############

try:
	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/' +  c.dataset + '/' + '/n_epochs_' + str(c.n_epochs))

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

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs - 1, i + 1, len(train_loader), flow.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, F_loss_meter, F_loss_meter)

			F_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_F = {
					'epoch': epoch,
					'model': flow.model.state_dict(),
					'optimizer': flow.optim.state_dict(),
					}
				save_checkpoint(checkpoint_F, log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), 'checkpoint_F_epoch_%03d' % (epoch))

			if c.test == True:
				size = 10000
			else:
				size = 300000

			with torch.no_grad():
				real = get_real_data(c.dataset, c.test, size)

				if c.weighted:
					inv, z = flow.model.sample(size)
					inv = inv.cpu().detach().numpy() * scales
				else:
					inv, z = flow.model.sample(size)
					inv = inv.cpu().detach().numpy() * scales

			distributions = Distribution(real, inv, 'epoch_%03d' % (epoch) + '_target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=False)
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
real = get_real_data(c.dataset, c.test, size)

# Save to hdf5
s1 = pd.HDFStore('augflow.h5')
s1.append('data', pd.DataFrame(fake))
s1.close()

# Make plots
distributions = Distribution(real, inv, 'HMC', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=False, weights=[], extra_data=fake)
distributions.plot()
