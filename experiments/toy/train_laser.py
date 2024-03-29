""" Train LaSeR """

# Basics
import sys, os
import torch
import numpy as np
import pandas as pd

# Models
from elsa.models.flow_model import INN
from elsa.models.gan_models import netD
from elsa.modules.mcmc import HamiltonMCMC

# Train utils
from elsa.utils.load_data import Loader
from elsa.utils.train_utils import (
    AverageMeter,
    print_log,
    get_real_data,
    save_checkpoint,
)

# Plotting
from elsa.utils.distributions import Distribution

# Load config and opts
import config_LSR as c
import opts

###########
## Setup ##
###########

opts.parse(sys.argv, c)
config_str = ""
config_str += "===" * 30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
    if v[0] == "_":
        continue
    s = eval("c.%s" % (v))
    config_str += " {:25}\t{}\n".format(v, s)

config_str += "===" * 30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(f"device: {device}")

###############
## Load data ##
###############

# TODO: Fix new scaler
train_loader, validate_loader, dataset_size, data_shape, scaler = Loader(
    c.datapath, c.dataset, c.batch_size, c.test, c.scale, c.weighted, device
)

if c.weighted:
    data_shape -= 1

print("\n" + "===" * 30 + "\n")

#######################
## Define Flow Model ##
#######################

flow = INN(
    in_dim=data_shape,
    aug_dim=c.aug_dim,
    n_blocks=c.n_blocks,
    n_units=c.n_units,
    n_layers=c.n_layers,
    device=device,
    config=c,
)
flow.define_model_architecture()  # This seems to be a bit annoying to call again?!
flow.set_optimizer()

print("\n" + "===" * 30 + "\n")
# print(flow.model)
print("Total parameters: %d" % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "===" * 30 + "\n")

################################
## Training primary generator ##
################################

try:
    log_dir = c.save_dir

    if not os.path.exists(
        log_dir + "/" + c.dataset + "/" + "hmc/n_epochs_" + str(c.n_epochs)
    ):
        os.makedirs(log_dir + "/" + c.dataset + "/" + "hmc/n_epochs_" + str(c.n_epochs))

    F_loss_meter = AverageMeter()

    print("Flow Training...")
    for epoch in range(c.n_epochs):
        for iteration in range(c.n_its_per_epoch_gen):

            i = 0

            for train_batch in train_loader:

                flow.model.train()
                flow.optim.zero_grad()

                f_loss = -flow.model.log_prob(train_batch.to(device)).mean()

                F_loss_meter.update(f_loss.item())

                f_loss.backward()
                flow.optim.step()

                i += 1

            if epoch == 0 or (epoch + 1) % c.show_interval == 0:
                print_log(
                    epoch + 1,
                    c.n_epochs,
                    i,
                    len(train_loader),
                    flow.scheduler.optimizer.param_groups[0]["lr"],
                    c.show_interval,
                    F_loss_meter,
                )

            F_loss_meter.reset()

        if (epoch + 1) % c.save_interval == 0 or epoch + 1 == c.n_epochs:
            if c.save_model == True:

                checkpoint_F = {
                    "epoch": epoch + 1,
                    "model": flow.model.state_dict(),
                    "optimizer": flow.optim.state_dict(),
                }
                save_checkpoint(
                    checkpoint_F,
                    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
                    "checkpoint_F_epoch_%03d" % (epoch + 1),
                )

            if c.test == True:
                size = 10000
            else:
                size = 300000

            with torch.no_grad():
                real = get_real_data(c.datapath, c.dataset, c.test, size)

                fake, _ = flow.model.sample(size)
                fake = scaler.inverse_transform(fake.cpu().detach().numpy())

            distributions = Distribution(
                real,
                fake,
                "epoch_%03d" % (epoch + 1) + "_target",
                "Flow",
                log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
                c.dataset,
            )
            distributions.plot()

        flow.scheduler.step()
except:
    if c.checkpoint_on_error:
        flow.model.save(c.filename + "_ABORT")
    raise

#######################
## Define Classifier ##
#######################

D = netD(
    in_dim=data_shape,
    num_layers=c.n_layers_disc,
    n_units=c.n_units_disc,
    device=device,
    config=c,
)
D.define_model_architecture_unreg()
D.set_optimizer()

print("\n" + "===" * 30 + "\n")
# print(D)
print("Total parameters: %d" % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "===" * 30 + "\n")

criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
phi_1 = lambda dreal, lreal, lfake: criterion_BCE(dreal, lreal)
phi_2 = lambda dfake, lreal, lfake: criterion_BCE(dfake, lfake)

D_loss_meter = AverageMeter()

#########################
## Training classifier ##
#########################

try:
    flow.model.eval()

    for epoch in range(c.n_epochs):
        for iteration in range(c.n_its_per_epoch_ref):

            i = 0

            for train_batch in train_loader:

                D.model.train()
                D.optim.zero_grad()

                label_real = torch.ones(c.batch_size).double().to(device)
                label_fake = torch.zeros(c.batch_size).double().to(device)

                d_result_real = D(train_batch).view(-1)
                d_loss_real_ = phi_1(d_result_real, label_real, None).mean(-1)

                fake, lat = flow.model.sample(c.batch_size)
                d_result_fake = D(fake).view(-1)
                d_loss_fake_ = phi_2(d_result_fake, None, label_fake).mean()
                d_loss = d_loss_real_ + d_loss_fake_
                D_loss_meter.update(d_loss.item())

                d_loss.backward()
                D.optim.step()

                i += 1

            if epoch == 0 or (epoch + 1) % c.show_interval == 0:
                print_log(
                    epoch + 1,
                    c.n_epochs,
                    i,
                    len(train_loader),
                    D.scheduler.optimizer.param_groups[0]["lr"],
                    c.show_interval,
                    D_loss_meter,
                )

            D_loss_meter.reset()

        if (epoch + 1) % c.save_interval == 0 or epoch + 1 == c.n_epochs:
            if c.save_model == True:

                checkpoint_D = {
                    "epoch": epoch + 1,
                    "model": D.model.state_dict(),
                    "optimizer": D.optim.state_dict(),
                }
                save_checkpoint(
                    checkpoint_D,
                    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
                    "checkpoint_D_epoch_%03d" % (epoch + 1),
                )

            if c.test == True:
                size = 10000
            else:
                size = 300000

            with torch.no_grad():
                real = get_real_data(c.datapath, c.dataset, c.test, size, scaler)

                noise = torch.randn(size, data_shape).detach().numpy()

                inv, lat = flow.model.sample(size)
                lat = lat.detach().numpy()

                enc = (
                    flow.model.encode(torch.Tensor(real).double().to(device))
                    .detach()
                    .numpy()
                )

                out_D = D(inv)
                weights = torch.exp(out_D).cpu().detach().numpy().flatten()

                inv = scaler.inverse_transform(inv.cpu().detach().numpy())

            distributions = Distribution(
                scaler.inverse_transform(real),
                inv,
                "epoch_%03d" % (epoch + 1) + "_target_weighted",
                "Weighted",
                log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
                c.dataset,
                latent=False,
                weights=weights,
                extra_data=inv,
            )
            distributions.plot()
            distributions = Distribution(
                enc,
                lat,
                "epoch_%03d" % (epoch + 1) + "_latent_weighted",
                "Weighted",
                log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
                c.dataset,
                latent=True,
                weights=weights,
            )
            distributions.plot()

        D.scheduler.step()

except:
    if c.checkpoint_on_error:
        D.save(c.filename + "_ABORT")
    raise

##############
## Sampling ##
##############

size = c.sample_size
N_CHAINS = 100

# Get refined samples
hamilton = HamiltonMCMC(
    flow, D, latent_dim=data_shape, L=30, eps=0.01, n_chains=N_CHAINS, burnin=5000
)
z, rate = hamilton.sample(data_shape, size // N_CHAINS)
print("rate = ", rate)

refined = flow.model.sample_refined(z)
refined = scaler.inverse_transform(refined.cpu().detach().numpy())
z = z.cpu().detach().numpy()

# get base samples and weights
fake, z_base = flow.model.sample(size)
out_D = D(fake)
weights = torch.exp(out_D)
z_base = z_base.cpu().detach().numpy()
weights = weights.cpu().detach().numpy()
fake = scaler.inverse_transform(fake.cpu().detach().numpy())

# get DCTR samples
dctr = np.concatenate((fake, z_base, weights), axis=-1)

# get real samples
real = get_real_data(c.datapath, c.dataset, c.test, size)


s1 = pd.HDFStore(
    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs) + "/" + "base.h5"
)
s2 = pd.HDFStore(
    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs) + "/" + "latent.h5"
)
s3 = pd.HDFStore(
    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs) + "/" + "refined.h5"
)
s4 = pd.HDFStore(
    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs) + "/" + "weighted.h5"
)

s1.append("data", pd.DataFrame(fake))
s2.append("data", pd.DataFrame(z))
s3.append("data", pd.DataFrame(refined))
s4.append("data", pd.DataFrame(dctr))

s1.close()
s2.close()
s3.close()
s4.close()

distributions = Distribution(
    real,
    refined,
    "HMC",
    "HMC",
    log_dir + "/" + c.dataset + "/hmc/n_epochs_" + str(c.n_epochs),
    c.dataset,
    extra_data=fake,
)
distributions.plot()
