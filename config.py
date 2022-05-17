######################
# Choose dataset: #
######################

dataset = '2d_4_gaussian_mixture_weighted'

#########
# Data: #
#########

weighted = False
scaler   = 1.

##############
# Training:  #
##############

lr = 1e-3
batch_size = 8000
gamma = 0.999
weight_decay = 0.
betas = (0.5, 0.9)

do_rev = False
do_fwd = True

n_epochs = 1000
n_its_per_epoch = 1

n_disc_updates = 5

#################
# Architecture: #
#################

wasserstein = True

n_blocks = 4
n_units  = 96
n_layers = 5

latent_dim_gen = 2

clamp = 0.01

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
progress_bar = True                         # Show a progress bar of each epoch

show_interval = 5
save_interval = 5

###################
# Loading/saving: #
###################

test = False
train = True

save_model = True
load_model = False

save_dir = './experiments'
checkpoint_on_error = False
