import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

rng = np.random.RandomState()

scale = 4.
batch_size = 1000000

centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
           (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]

centers = [(scale * x, scale * y) for x, y in centers]

dataset = []
for i in range(batch_size):
	point = rng.randn(2) * 0.5
	idx = rng.randint(8)
	center = centers[idx]
	point[0] += center[0]
	point[1] += center[1]
	dataset.append(point)

dataset = np.array(dataset)
dataset /= 1.414

s = pd.HDFStore('eight_gaussians.h5')
s.append('data', pd.DataFrame(dataset))
s.close()
