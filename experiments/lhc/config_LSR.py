##################
# Choose dataset #
##################

datapath = '../../datasets/lhc/'
dataset = 'wp_3j' # wp_2j, wp_3j
e_had = 14000
masses = [80.419] + [0.] * 3
ptcuts = [0.] + [20.0] * 3

########
# Data #
########

scale = None
sample_size = int(1e6)
gen_scaler = "Rambo" # Simple, Schumann, Momenta, Minrep, Heimel, Rambo
disc_scaler = "Laser" # Laser and above

#################
# Flow Training #
#################

n_epochs = 51
batch_size = 2000
betas = (0.9, 0.999)

lr_scheduler = "onecycle" # or "exponential"
lr = 5e-4
max_lr = 3e-3

gamma = 0.995
exp_decay = 0.1
weight_decay = 1e-5

#######################
# Classifier Training #
#######################

disc_n_epochs = 50

disc_lr_scheduler = "onecycle" # or "exponential"
disc_lr = 5e-4
disc_max_lr = 5e-3

disc_gamma = 0.995
disc_exp_decay = 0.1
n_its_per_epoch = 1

################
# Architecture #
################

coupling = "rqs"
n_blocks = 14
n_units  = 80
n_layers = 2
aug_dim  = 0 

disc_n_units  = 128 #make bigger and better?
disc_n_layers = 10

###################
# Logging/preview #
###################

show_interval = 1

##################
# Loading/saving #
##################

test = False
train = True

save_model = True
save_dir = './results'
