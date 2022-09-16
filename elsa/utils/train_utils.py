""" Train utils """

import torch
import pandas as pd
import numpy as np
from ..modules.preprocess import Scaler


class AverageMeter:
    """Computes and stores the average and current value of the loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(epoch, epochs, iteration, iters, learning_rate, display, loss):

    print(f"epoch: [{epoch}/{epochs}], iteration: [{iteration}/{iters}],  Learning rate: {learning_rate}")
    print("Loss = {loss.val:.8f} (ave = {loss.avg:.8f})".format(display, loss=loss))

def save_checkpoint(state, filename="checkpoint"):
    torch.save(state, filename + ".pth.tar")


############


def remove_momenta(x):

    x = np.delete(x, np.s_[3:5], axis=1)

    return x


def add_momenta(x):

    px = -x[:, [0]]
    py = -x[:, [1]]

    if torch.is_tensor(x) == True:
        p = torch.cat((px, py), -1)
        x = torch.cat(((x[:, :3], p, x[:, 3:])), dim=1)
    else:
        p = np.concatenate((px, py), -1)
        x = np.hstack((x[:, :3], p, x[:, 3:]))
    return x


def remove_energies(x, n_particles=2):

    momenta = []

    for p in range(n_particles):
        px = 4 * p + 1
        pz = 4 * p + 3

        momenta.extend([i for i in range(px, pz)])

    momenta = x[:, momenta]

    return momenta


def add_energies(x, masses=[0.0, 0.0]):

    """
    add energies to 3 vectors
    """
    n_particles = len(masses)
    masses = np.array(masses)

    particles = []

    for p in range(n_particles):
        momenta = x[:, 3 * p : 3 * p + 3]
        momenta2 = torch.sum(torch.mul(momenta, momenta), dim=-1).float()

        mass = torch.Tensor(momenta2.shape).fill_(masses[p] ** 2)

        E = mass.add(momenta2)
        E = torch.sqrt(E).clone()
        E = E.unsqueeze_(-1)
        E = torch.cat((E, momenta), dim=-1)

        particles.append(E)

    particles = torch.cat(particles, dim=1)

    return particles


def save_checkpoint(state, log_dir, name):
    path = log_dir + "/" + name + ".pth"
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, topo=False):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    if topo:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]
    else:
        model.model.load_state_dict(checkpoint["model"])
        model.optim.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]


def get_real_data(
    datapath: str,
    dataset: str,
    test: bool,
    sample_size: int,
    scaler: Scaler = None,
):

    data = pd.read_hdf(datapath + dataset + ".h5").values
    if scaler is not None:
        data = scaler.transform(data)

    if test == True:
        split = int(len(data) * 0.1)
    else:
        split = int(len(data) * 0.8)

    data = data[:split]

    index = np.arange(data.shape[0])

    if sample_size <= data.shape[0]:
        choice = np.random.choice(index, sample_size, replace=False)
    else:
        choice = np.random.choice(index, sample_size, replace=True)

    data = data[choice]

    return data


def preprocess(x, inverse=False, masses=[]):

    if inverse:
        n_particles = x.shape[1] // 3
    else:
        n_particles = x.shape[1] // 4

        # remove energies
    if inverse:
        assert len(masses) == n_particles
        x = add_energies(x, masses=masses)
    else:
        x = remove_energies(x, n_particles=n_particles)

    # what else to do?

    return x


def get_masses(x, topologies=[[0, 1]]):

    res = len(topologies)

    m = torch.tensor((), dtype=torch.float)

    for i in range(res):

        Es = torch.zeros(x.shape[0], dtype=torch.float)
        Pxs = torch.zeros(x.shape[0], dtype=torch.float)
        Pys = torch.zeros(x.shape[0], dtype=torch.float)
        Pzs = torch.zeros(x.shape[0], dtype=torch.float)

        for p in topologies[i]:
            Es += x[:, 0 + p * 4]
            Pxs += x[:, 1 + p * 4]
            Pys += x[:, 2 + p * 4]
            Pzs += x[:, 3 + p * 4]

        m2 = (
            torch.mul(Es, Es)
            - torch.mul(Pxs, Pxs)
            - torch.mul(Pys, Pys)
            - torch.mul(Pzs, Pzs)
        )
        m2 = m2.float()
        m = torch.cat((m, torch.sqrt(torch.clamp(m2, min=1e-20))))

    m = m.view(res, int(m.shape[0] / res))

    return m


def get_masses_squared(x, topologies=[[0, 1]]):

    res = len(topologies)

    m = torch.tensor((), dtype=torch.float)

    for i in range(res):

        Es = torch.zeros(x.shape[0], dtype=torch.float)
        Pxs = torch.zeros(x.shape[0], dtype=torch.float)
        Pys = torch.zeros(x.shape[0], dtype=torch.float)
        Pzs = torch.zeros(x.shape[0], dtype=torch.float)

        for p in topologies[i]:
            Es += x[:, 0 + p * 4]
            Pxs += x[:, 1 + p * 4]
            Pys += x[:, 2 + p * 4]
            Pzs += x[:, 3 + p * 4]

        m2 = (
            torch.mul(Es, Es)
            - torch.mul(Pxs, Pxs)
            - torch.mul(Pys, Pys)
            - torch.mul(Pzs, Pzs)
        )
        m2 = m2.float()
        m = torch.cat((m, m2))

    m = m.view(res, int(m.shape[0] / res))

    return m
