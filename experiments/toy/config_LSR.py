##################
# Choose dataset #
##################

datapath = '../../datasets/toy/'
dataset = 'pinwheel' # eight_gaussians, pinwheel, circles

########
# Data #
########

weighted = False
scaler   = 1.

############
# Training #
############

lr = 1e-3
lr_ref = 1e-3

batch_size = 1024
gamma = 0.995
weight_decay = 1e-5

betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 100
n_its_per_epoch_gen = 1
n_its_per_epoch_ref = 1

n_disc_updates = 5

mmd = False

################
# Architecture #
################

n_blocks = 8
n_units  = 32
n_layers = 2
aug_dim  = 0 

n_units_disc  = 80
n_layers_disc = 8

latent_dim_gen = 2 #for LSRGAN

###################
# Logging/preview #
###################

loss_names = ['L', 'L_rev']
progress_bar = True

show_interval = 5
save_interval = 50

##################
# Loading/saving #
##################

test = False
train = True

save_model = True
load_model = False

save_dir = './results'
checkpoint_on_error = False
