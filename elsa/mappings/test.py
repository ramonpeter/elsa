from elsa.mappings.rambo import RamboOnDietHadron
from elsa.utils.observables import Observable
import numpy as np
import sys
import matplotlib.pyplot as plt

# Test if it gives some sensible momenta

E_HAD = 14000
NPARTICLES = 3
DIMS = 3 * NPARTICLES - 2
MASSES = [90.0,0.0,0.0] # in GeV

ps_generator = RamboOnDietHadron(E_HAD, NPARTICLES, MASSES)

x = np.random.rand(1,DIMS)
p = ps_generator.map(x)


obs = Observable()
l = [obs.invariant_mass(p, [i])[..., None] for i in range(NPARTICLES)]
masses = np.concatenate(l, axis=-1)
print(f"masses: {masses}")

count = p.size // (NPARTICLES * 4)
p = p.reshape((count, NPARTICLES, 4))
# print(p)
# #print(p.shape)
print(f"CM-momentum: {np.sum(p, 1)}")

p = ps_generator.map(x)
y = ps_generator.map_inverse(p)
print(f"x = {x}")
print(f"y = {y}")


# Plot testing

# tau0= 80/14000
# n=1000000
# z = np.random.rand(n,1)
# tau = (1-z)/(1+z)*14000
# logtau2 = z * np.log(tau0)
# tau2 = np.exp(logtau2) * 14000

# y_t, x_t = np.histogram(tau , 50, density=True, range=[0,2000])#, weights=weights)
# y_t2, x_t2 = np.histogram(tau2 , 50, density=True, range=[0,2000])#, weights=weights)
# fig, axs = plt.subplots(1, 1)
# #axs.step(x_t[:100], y_t, label='Truth', linewidth=1.0, where='mid')
# axs.step(x_t2[:50], y_t2, label='Log', linewidth=1.0, where='mid')
# axs.axvline(tau0*14000, linewidth=1, linestyle='--', color='grey')
# #axs.set_yscale('log')
# #axs.set_ylim((0,100))
# fig.savefig('tau.pdf', bbox_inches='tight', format='pdf', pad_inches=0.05)
# plt.close()


