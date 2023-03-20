""" Train augmented/baseline flow """

# Basics
import sys, os
import torch
import numpy as np
import pandas as pd
import time

# Model
from elsa.models.flow_model import AffineFlow, CubicFlow, RQSFlow

# Train utils
from elsa.utils.load_data import Loader
from elsa.utils.train_utils import (
    AverageMeter,
    print_log,
    get_real_data,
)

# Plotting
from elsa.utils.distributions import Distribution

# Load config and opts
import config_aug as conf
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
    flow = RQSFlow(
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
# print(flow.model)
print("Total parameters: %d" % sum([np.prod(p.size()) for p in flow.params_trainable]))
print("\n" + "===" * 30 + "\n")

##############
## Training ##
##############

# define log_dir
model = "augflow" if conf.aug_dim > 0 else "baseflow"
log_dir = f"{conf.save_dir}/{conf.dataset}/{model}/{conf.n_epochs:03d}_epochs_{conf.coupling}_{conf.gen_scaler}"
model_path = os.path.join(log_dir, "model.pth")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if conf.train:
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
        print("Save Model...")
        flow.save(model_path)

else:
    print("Load Model...")
    flow.load(model_path)
    print("Model loaded successfully")


##############
## Sampling ##
##############

size = conf.sample_size

# Get flow samples
fake, _ = flow.model.sample(size)
fake = fake.cpu().detach().numpy()
fake = loader.gen_scaler.inverse_transform(fake)

# Get real samples
real = get_real_data(conf.datapath, conf.dataset, conf.test, size)

# Save to hdf5
s1 = pd.HDFStore(f"{log_dir}/{model}.h5")
s1.append("data", pd.DataFrame(fake))
s1.close()

# Make plots
distributions = Distribution(real, fake, f"{model}", f"{model}", log_dir, conf.dataset)
distributions.plot()
