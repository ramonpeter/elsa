from elsa.utils.train_utils import *
from elsa.utils.plotting.distributions import *
from elsa.utils.plotting.plots import *
from elsa.load_data import *

from survae_model import INN
from GAN_models import netD
from elsa.mcmc import HamiltonMCMC

import sys, os

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

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.weighted)
scales_tensor = torch.Tensor(scales).double().to(device)

if c.weighted:
	data_shape -= 1

print("\n" + "==="*30 + "\n")

flow = INN(in_dim=data_shape, aug_dim=c.aug_dim, n_blocks=c.n_blocks, internal_size=c.n_units, n_layers=c.n_layers, init_zeros=False, dropout=False)
flow.define_model_architecture()
flow.set_optimizer()

print("\n" + "==="*30 + "\n")
print(flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "==="*30 + "\n")

# Train primary generator
try:
	log_dir = c.save_dir

	if not os.path.exists(log_dir + '/' + c.dataset + '/' + '/n_epochs_' + str(c.n_epochs)):
		os.makedirs(log_dir + '/' +  c.dataset + '/' + '/n_epochs_' + str(c.n_epochs))

	F_loss_meter = AverageMeter()

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
					#inv = model.sample(size).cpu().detach().numpy() * scales
					inv, z = flow.model.sample(size)
					inv = inv.cpu().detach().numpy() * scales

			distributions = Distribution(real, inv, 'epoch_%03d' % (epoch) + '_target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=False)
			distributions.plot()

		flow.scheduler.step()
except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 

'''
size = 1000000
fake, z_ = flow.model.sample(size)

fake = fake.cpu().detach().numpy()
fake *= scales

s1 = pd.HDFStore('base.h5')

s1.append('data', pd.DataFrame(fake))

s1.close()
sys.exit()
'''

D = netD(in_dim=data_shape, num_layers=c.n_layers_disc, internal_size=c.n_units_disc)
D.define_model_architecture_unreg()
D.set_optimizer()

print("\n" + "==="*30 + "\n")
print(D)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "==="*30 + "\n")

criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
phi_1 = lambda dreal, lreal, lfake: criterion_BCE(dreal, lreal)
phi_2 = lambda dfake, lreal, lfake: criterion_BCE(dfake, lfake)

D_loss_meter = AverageMeter()

# Train refiner
try:
	flow.model.eval()

	for epoch in range(c.n_epochs):
	#for epoch in range(int(c.n_epochs / 2) + 1):
		for iteration in range(c.n_its_per_epoch_ref):

			i=0

			for data in train_loader:

				data /= scales_tensor

				D.model.train()
				D.optim.zero_grad()

				label_real = torch.ones(c.batch_size).double().to(device)
				label_fake = torch.zeros(c.batch_size).double().to(device)

				d_result_real = D(data).view(-1)
				d_loss_real_ = phi_1(d_result_real, label_real, None).mean(-1)
		
				fake, lat = flow.model.sample(c.batch_size)
				d_result_fake = D(fake).view(-1)
				d_loss_fake_ = phi_2(d_result_fake, None, label_fake).mean()
				d_loss = d_loss_real_ + d_loss_fake_
				D_loss_meter.update(d_loss.item())

				d_loss.backward()
				D.optim.step()

				i += 1

			if epoch == 0 or epoch % c.show_interval == 0:
				print_log(epoch, c.n_epochs - 1, i + 1, len(train_loader), D.scheduler.optimizer.param_groups[0]['lr'],
							   c.show_interval, D_loss_meter, D_loss_meter)

			D_loss_meter.reset()

		if epoch % c.save_interval == 0 or epoch + 1 == c.n_epochs:
			if c.save_model == True:

				checkpoint_D = {
					'epoch': epoch,
					'model': D.model.state_dict(),
					'optimizer': D.optim.state_dict(),
					}
				save_checkpoint(checkpoint_D, log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), 'checkpoint_D_epoch_%03d' % (epoch))

			if c.test == True:
				size = 10000
			else:
				size = 300000

			with torch.no_grad():
				real = get_real_data(c.dataset, c.test, size)
				noise = torch.randn(size, data_shape).detach().numpy()

				inv, lat = flow.model.sample(size)
				lat = lat.detach().numpy()

				enc = flow.model.encode(torch.Tensor(real / scales)).detach().numpy()

				out_D = D(inv)
				weights = torch.exp(out_D).cpu().detach().numpy().flatten()

				inv = inv.cpu().detach().numpy() * scales

			distributions = Distribution(real, inv, 'epoch_%03d' % (epoch) + '_target_weighted', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=False, weights=weights, extra_data = inv)
			distributions.plot()
			distributions = Distribution(enc, lat, 'epoch_%03d' % (epoch) + '_latent_weighted', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=True, weights=weights)
			distributions.plot()

		D.scheduler.step()

except:
	if c.checkpoint_on_error:
		model.save(c.filename + '_ABORT')
	raise 

# Generate unweighted optimal latent space

hamilton = HamiltonMCMC(flow.model, D, latent_dim=data_shape, L=30, eps=0.01, n_chains=100, burnout=5000)
z, rate = hamilton.sample(data_shape, 10000)

print('rate = ', rate)

inv = flow.model.sample_custom(z)
inv = inv.cpu().detach().numpy() * scales

size = 1000000

fake, z_ = flow.model.sample(size)

out_D = D(fake)
weights = torch.exp(out_D)

#fake = fake.cpu().detach().numpy()
real = get_real_data(c.dataset, c.test, size)
dctr = torch.cat((fake, z_, weights), -1).cpu().detach().numpy()

fake = fake.cpu().detach().numpy()
fake *= scales

s1 = pd.HDFStore('base.h5')
s2 = pd.HDFStore('latent.h5')
s3 = pd.HDFStore('refined.h5')
s4 = pd.HDFStore('weighted.h5')

s1.append('data', pd.DataFrame(fake))
s2.append('data', pd.DataFrame(z.cpu().detach().numpy()))
s3.append('data', pd.DataFrame(inv))
s4.append('data', pd.DataFrame(dctr))

s1.close()
s2.close()
s3.close()
s4.close()

distributions = Distribution(real, inv, 'HMC', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, latent=False, weights=[], extra_data=fake)
distributions.plot()
