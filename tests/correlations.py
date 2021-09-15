import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import numpy as np
import proplot as pplt

from src.mas import cic_mas
from src.correlations import powspec, powspec_vec

import MAS_library as MASL
import Pk_library as PKL


n_bins = 32
n_part = 1000
box_size = 400.
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
particles = box_size * jax.random.uniform(subkey, (n_part, 4))



#
#particles = np.loadtxt("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), dtype=np.float32)
#mask = ((particles < box_size) & (particles > 0)).all(axis=1)
#particles = jax.device_put(particles[mask].copy())
#n_part = particles.shape[0]
#print(n_part)





w = jnp.ones(n_part)
s = time.time()
delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas(delta,
                particles[:,0], particles[:,1], particles[:,2], w, 
                n_part, 
                particles[:,0].min(), particles[:,1].min(), particles[:,2].min(),
                box_size,
                n_bins,
                True)
delta /= delta.mean()
delta -= 1.
print(f"MAS took {time.time() - s} s.", flush=True)

delta_pl = np.zeros((n_bins,n_bins,n_bins), dtype=np.float32)
MASL.MA(np.array(particles[:,:3]).astype(np.float32), delta_pl, box_size, 'CIC', verbose=True)
delta_pl /= delta_pl.mean()
delta_pl -= 1.





fig, ax = pplt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
ax[0].imshow(delta[:,:,1], colorbar='r')
ax[1].imshow(delta_pl[:,:,1], colorbar='r')

fig.savefig("plots/jax-cic.png", dpi=300)

k, pk, phase, modes = powspec(delta, box_size, 2)
ax[2].plot(k, pk[:,0], label='loops')

Pk = PKL.Pk(delta_pl, box_size, 2, 'CIC', 8)
mask = Pk.k3D < 1.2
k = Pk.k3D[mask]
pk = Pk.Pk[mask]


ax[2].plot(k, pk[:,0], label='Pylians', ls = '--')

k, pk, phase, modes = powspec_vec(delta, box_size, 2) 
ax[2].plot(k, pk[:,0], label='vector')
ax[2].legend()

CF = PKL.Xi(delta_pl, box_size, 'CIC', 2, 8)
mask = CF.r3D < 200
r = CF.r3D[mask]
xi = CF.xi[mask]
ax[3].plot(r, r**2*xi[:,0], label='Pylians', ls = '--')



ax[2].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[3].format(xlabel='$s$ [Mpc / $h$]', ylabel=r'$s^2\xi$')
ax[0].format(title='$\delta(x)$')
ax[1].format(title='$\delta(x)$ Pylians')

fig.savefig("plots/jax-cic.png", dpi=300)



