""" Train LaSeR """

# Basics
import sys, os
import torch
import numpy as np
import pandas as pd
import time

# Models
from elsa.models.flow_model import AffineFlow, CubicFlow, RQSFlow, RealRQSFlow
from elsa.models.gan_models import netD
from elsa.modules.mcmc import HamiltonMCMC

# Train utils
from elsa.utils.train_utils import (
    AverageMeter,
    print_log,
    get_real_data,
)
from elsa.utils.load_data import Loader

# Plotting
from elsa.utils.distributions import Distribution

# Load config and opts
import config_LSR as conf
import opts

###########
## Setup ##
###########

opts.parse(sys.argv, conf)
config_str = ""
config_str += "===" * 30 + "\n"
config_str += "Config options:\n\n"

for v in dir(conf):
    if v[0] == "_":
        continue
    s = eval("conf.%s" % (v))
    config_str += " {:25}\t{}\n".format(v, s)

config_str += "===" * 30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(f"device: {device}")

###############
## Load data ##
###############

loader = Loader(conf, device)
STEPS_PER_EPOCH = len(loader.train_data)

print("\n" + "===" * 30 + "\n")

#######################
## Define Flow Model ##
#######################

if conf.coupling == "affine":
    flow = AffineFlow(
        in_dim=loader.shape,
        aug_dim=conf.aug_dim,
        config=conf,
        unit_hypercube=loader.is_hypercube,
        steps_per_epoch=STEPS_PER_EPOCH,
        device=device,
    )
elif conf.coupling == "rqs":
    flow = RealRQSFlow( # Real for HMC with Rambo
        in_dim=loader.shape,
        aug_dim=conf.aug_dim,
        config=conf,
        unit_hypercube=loader.is_hypercube,
        steps_per_epoch=STEPS_PER_EPOCH,
        device=device,
    )
elif conf.coupling == "cubic":
    flow = CubicFlow(
        in_dim=loader.shape,
        aug_dim=conf.aug_dim,
        config=conf,
        unit_hypercube=loader.is_hypercube,
        steps_per_epoch=STEPS_PER_EPOCH,
        device=device,
    )
else:
    raise ValueError(f"Coupling {conf.coupling} is not a valid argument")

flow.define_model_architecture()
flow.set_optimizer()

print("\n" + "===" * 30 + "\n")
#print(flow.model)
print("Total parameters: %d" % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "===" * 30 + "\n")

################################
## Training primary generator ##
################################

# define log_dir
log_dir = f"{conf.save_dir}/{conf.dataset}/hmc/{conf.n_epochs:03d}_fepochs_{conf.coupling}_f{conf.gen_scaler}_d{conf.disc_scaler}"
model_path = os.path.join(log_dir, "flow_model.pth")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if conf.train and conf.train_flow:
    F_loss_meter = AverageMeter()
    print("Training...")
    start_time = time.time()
    for epoch in range(conf.n_epochs):
        i = 0

        for train_batch in loader.train_data:
            flow.model.train()
            flow.optim.zero_grad()
            f_loss = -flow.model.log_prob(train_batch.to(device)).mean()

            F_loss_meter.update(f_loss.item())

            f_loss.backward()
            flow.optim.step()
            if flow.scheduler is not None:
                flow.scheduler.step()

            i += 1

        if epoch == 0 or (epoch + 1) % conf.show_interval == 0:
            print_log(
                epoch + 1,
                conf.n_epochs,
                i,
                len(loader.train_data),
                flow.scheduler.optimizer.param_groups[0]["lr"],
                conf.show_interval,
                F_loss_meter,
            )

            F_loss_meter.reset()

    train_time = time.time() - start_time
    print(f"--- Run time: {train_time / 60 / 60} hour ---")
    print(f"--- Run time: {train_time / 60} mins ---")
    print(f"--- Run time: {train_time} secs ---")
    if conf.save_model == True:
        print("Save Flow Model...")
        flow.save(model_path)

else:
    print("Load Flow Model...")
    flow.load(model_path)

#######################
## Define Classifier ##
#######################

# TODO: Add an augmentation layer to make life easier! Maybe not needed working on hypercube?
D = netD(
    in_dim=loader.disc_shape,
    config=conf,
    steps_per_epoch=STEPS_PER_EPOCH,
    device=device,
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

#
d_model_path = os.path.join(log_dir, "d_model.pth")

if conf.train and conf.train_disc:
    start_time = time.time()
    flow.model.eval() # make sure its not trained!
    for epoch in range(conf.disc_n_epochs):
        for iteration in range(conf.n_its_per_epoch):

            i = 0

            for train_batch in loader.disc_train_data:

                D.model.train()
                D.optim.zero_grad()

                label_real = torch.ones(conf.batch_size).double().to(device)
                label_fake = torch.zeros(conf.batch_size).double().to(device)

                d_result_real = D(train_batch.to(device)).view(-1)
                d_loss_real_ = phi_1(d_result_real, label_real, None).mean()

                fake, _ = flow.model.sample(conf.batch_size)
                if conf.disc_scaler is not None:
                    fake = loader.gen_scaler.inverse_transform(fake.cpu().detach().numpy())
                    fake = loader.disc_scaler.transform(fake)
                    fake = torch.tensor(fake)
                d_result_fake = D(fake.to(device)).view(-1)
                d_loss_fake_ = phi_2(d_result_fake, None, label_fake).mean()
                d_loss = d_loss_real_ + d_loss_fake_
                D_loss_meter.update(d_loss.item())

                d_loss.backward()
                D.optim.step()
                if D.scheduler is not None:
                    D.scheduler.step()

                i += 1

            if epoch == 0 or (epoch + 1) % conf.show_interval == 0:
                print_log(
                    epoch + 1,
                    conf.disc_n_epochs,
                    i,
                    len(loader.disc_train_data),
                    D.scheduler.optimizer.param_groups[0]["lr"],
                    conf.show_interval,
                    D_loss_meter,
                )

            D_loss_meter.reset()

    train_time = time.time() - start_time
    print(f"--- Run time: {train_time / 60 / 60} hour ---")
    print(f"--- Run time: {train_time / 60} mins ---")
    print(f"--- Run time: {train_time} secs ---")
    if conf.save_model == True:
        print("Save Classifier Model...")
        D.save(d_model_path)

else:
    print("Load Classifier Model...")
    D.load(d_model_path)

##############
## Sampling ##
##############

print("Sampling...")
size = conf.sample_size

if conf.hmc:
    N_CHAINS = 100

    # Get refined samples
    hamilton = HamiltonMCMC(
        flow, D, loader, latent_dim=loader.shape, L=30, eps=0.01, n_chains=N_CHAINS, burnin=500,
    )
    z, rate = hamilton.sample(loader.shape, size // N_CHAINS)
    print("rate = ", rate)

    refined = flow.model.sample_refined(z)
    refined = loader.gen_scaler.inverse_transform(refined.cpu().detach().numpy())
    z = z.cpu().detach().numpy()
    
    # save refined latent
    s3 =  pd.HDFStore(f"{log_dir}/latent.h5")
    s3.append("data", pd.DataFrame(z))
    s3.close()
    
    # save refined feature space
    s4 =  pd.HDFStore(f"{log_dir}/refined.h5")
    s4.append("data", pd.DataFrame(refined))  
    s4.close()


# get base samples and weights
print("Get base samples and weights...")
fake, z_base = flow.model.sample(size)
if conf.disc_scaler is not None:
    fake = loader.gen_scaler.inverse_transform(fake.cpu().detach().numpy())
    fake_reparam = np.copy(fake)
    fake_reparam = loader.disc_scaler.transform(fake_reparam)
    fake_reparam = torch.tensor(fake_reparam)
else:
    fake_reparam = np.copy(fake)
out_D = D(fake_reparam)
weights = torch.exp(out_D)
z_base = z_base.cpu().detach().numpy()
weights = weights.cpu().detach().numpy()
if conf.disc_scaler is None:
    fake = loader.gen_scaler.inverse_transform(fake.cpu().detach().numpy())


# get DCTR samples
print("Pack DCTR samples...")
dctr = np.concatenate((fake_reparam.cpu().detach().numpy(), z_base, weights), axis=-1)

# # get real samples
print("Get real samples...")
real = get_real_data(conf.datapath, conf.dataset, conf.test, size)

# Save to hdf5
s1 =  pd.HDFStore(f"{log_dir}/base.h5")
s2 =  pd.HDFStore(f"{log_dir}/dctr.h5")

s1.append("data", pd.DataFrame(fake))
s2.append("data", pd.DataFrame(dctr))

s1.close()
s2.close()

# Make plots
print("Make the plots...")
distributions = Distribution(
    real,
    fake,
    "baseflow",
    "baseflow",
    log_dir,
    conf.dataset,
    weights=weights,
)
distributions.plot()
