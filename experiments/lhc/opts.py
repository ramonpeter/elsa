import argparse

def parse(args, config):

	parser = argparse.ArgumentParser(prog=args[0])

	parser.add_argument('-l', '--lr',            default=config.lr, dest='lr', type=float)
	parser.add_argument('-d', '--gamma',         default=config.gamma, dest='gamma', type=float)

	parser.add_argument('-b', '--batch_size',    default=config.batch_size, dest='batch_size', type=int)
	parser.add_argument('-n', '--n_iterations',  default=config.n_its_per_epoch_gen, dest='n_its_per_epoch_gen', type=int)
	parser.add_argument('-N', '--epochs',        default=config.n_epochs, dest='n_epochs', type=int)

	opts = parser.parse_args(args[1:])

	config.lr                  = opts.lr
	config.batch_size          = opts.batch_size
	config.gamma               = opts.gamma
	config.n_its_per_epoch_gen = opts.n_its_per_epoch_gen
	config.n_epochs        	   = opts.n_epochs
