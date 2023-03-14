##################
# Choose dataset #
##################

datapath = '../../datasets/lhc/'
dataset = 'wp_3j' # wp_2j, wp_3j, wp_4j

########
# Data #
########

weighted = False
scale   = None
sample_size = 1000000

############
# Training #
############

lr = 4e-4
lr_ref = 1e-3
max_lr = 3e-3

batch_size = 2000 # maybe bigger batches?
gamma = 0.995
weight_decay = 1e-5

betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 50
n_its_per_epoch_gen = 1

################
# Architecture #
################

n_blocks = 14 # was 14
n_units  = 64 # was 80
n_layers = 3 # was 2
aug_dim  = 0 # dims are 7/10/13 for 2j/3j/4j

###################
# Logging/preview #
###################

loss_names = ['L', 'L_rev']
progress_bar = True

show_interval = 1 # 10
save_interval = 50 # 20

##################
# Loading/saving #
##################

test = False
train = True

save_model = True
load_model = False

save_dir = './results'
checkpoint_on_error = False
