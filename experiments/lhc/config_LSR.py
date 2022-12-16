##################
# Choose dataset #
##################

datapath = '../../datasets/lhc/'
dataset = 'wp_3j' # wp_2j, wp_3j, wp_4j

########
# Data #
########

weighted = False
scale = None
sample_size = 1000000

############
# Training #
############

lr = 1e-4
lr_ref = 1e-3
max_lr = 5e-3

batch_size = 2000
gamma = 0.995
weight_decay = 1e-5

betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 50
n_its_per_epoch_gen = 1
n_its_per_epoch_ref = 1

n_disc_updates = 5

mmd = False

################
# Architecture #
################

n_blocks = 14
n_units  = 128
n_layers = 3
aug_dim  = 0 

n_units_disc  = 64 #make bigger and better?
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
