##################
# Choose dataset #
##################

datapath = '../../datasets/lhc/'
dataset = 'wp_2j' # wp_2j, wp_3j
e_had = 14000
masses = [80.419] + [0.] * 2
ptcuts = [0.] + [20.0] * 2

######################
# Data preprocessing #
######################

scale = None
sample_size = int(1e6)
gen_scaler = "Rambo" # Simple, Schumann, Momenta, Minrep, Heimel, Ramobo
disc_scaler = None

############
# Training #
############

n_epochs = 5
batch_size = 2000
betas = (0.9, 0.999)

lr_scheduler = "onecycle" # or "exponential"
lr = 5e-4
max_lr = 3e-3

gamma = 0.995
exp_decay = 0.1
weight_decay = 1e-5

################
# Architecture #
################

coupling = "affine"
n_blocks = 14 # was 14
n_units  = 80 # was 80
n_layers = 2 # was 2
aug_dim  = 0 # dims are 7/10/13 for 2j/3j/4j

###################
# Logging/preview #
###################

show_interval = 1 # how often print the loss

##################
# Loading/saving #
##################

test = False
train = True

save_model = True
save_dir = './results'
