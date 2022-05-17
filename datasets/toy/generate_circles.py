import numpy as np
import pandas as pd

import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

rng = np.random.RandomState()

n_samples1 = n_samples4 = n_samples3 = n_samples2 = 250000

linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)

circ4_x =        np.cos(linspace4)
circ3_x = 0.75 * np.cos(linspace3)
circ2_x = 0.5 *  np.cos(linspace2)
circ1_x = 0.25 * np.cos(linspace1)
circ4_y =        np.sin(linspace4)
circ3_y = 0.75 * np.sin(linspace3)
circ2_y = 0.5 *  np.sin(linspace2)
circ1_y = 0.25 * np.sin(linspace1)

X = np.vstack([np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]), np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])]).T * 3.0
X = util_shuffle(X, random_state=rng)
X = X + rng.normal(scale=0.08, size=X.shape)

s = pd.HDFStore('circles.h5')
s.append('data', pd.DataFrame(X))
s.close()

