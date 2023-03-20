import argparse


def parse(args, config):

    parser = argparse.ArgumentParser(prog=args[0])

    parser.add_argument("--lr", default=config.lr, dest="lr", type=float)
    parser.add_argument("--gamma", default=config.gamma, dest="gamma", type=float)
    parser.add_argument("--epochs", default=config.n_epochs, dest="n_epochs", type=int)
    parser.add_argument(
        "--batch_size",
        default=config.batch_size,
        dest="batch_size",
        type=int,
    )
    parser.add_argument(
        "--coupling",
        type=str,
        default=config.coupling,
        choices={"affine", "rqs", "cubic"},
    )

    opts = parser.parse_args(args[1:])

    config.lr = opts.lr
    config.batch_size = opts.batch_size
    config.gamma = opts.gamma
    config.n_epochs = opts.n_epochs
    config.n_epochs = opts.n_epochs
    config.coupling = opts.coupling
