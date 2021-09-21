import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import numpy as np
import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental

import MAS_library as MASL
import Pk_library as PKL


n_bins = 300
n_part = 1000
box_size = 2500.
k_ny = jnp.pi * n_bins / box_size
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
particles = box_size * jax.random.uniform(subkey, (n_part, 4))



particles = np.loadtxt("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), dtype=np.float32)
mask = ((particles < box_size) & (particles > 0)).all(axis=1)
particles = jax.device_put(particles[mask].copy())
n_part = particles.shape[0]
print(n_part)


shot_noise = box_size**3 / n_part


w = jnp.ones(n_part)
s = time.time()
delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
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
ax[0].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[1].imshow(delta_pl[:,:,:].mean(axis=2), colorbar='r')

fig.savefig("plots/jax-cic.png", dpi=300)


Pk = PKL.Pk(delta_pl, box_size, 2, 'CIC', 8)
mask = Pk.k3D < k_ny
k = Pk.k3D[mask]
pk = Pk.Pk[mask]
ax[2].plot(k, pk[:,0]-shot_noise, label='Pylians', ls = '--')


kedges = jnp.arange(1e-4, 5, 0.2e-2)

k, pk, modes = powspec_vec(delta, box_size, kedges) 
mask = k < k_ny
k = k[mask]
pk = pk[mask]
#k, pk, modes = powspec_vec_fundamental(delta, box_size) 
ax[2].plot(k, pk[:,0]-shot_noise, label='vector')
ax[2].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
fig.savefig("plots/jax-cic.png", dpi=300)


CF = PKL.Xi(delta_pl, box_size, 'CIC', 2, 8)
mask = CF.r3D < 200
r = CF.r3D[mask]
xi = CF.xi[mask]
ax[3].plot(r, r**2*xi[:,0], label='Pylians', ls = '--')

r, xi, modes = xi_vec(delta, box_size, kedges) 
#r, xi, modes = xi_vec_fundamental(delta, box_size) 
mask = r < 200
r = r[mask]
xi = xi[mask]
ax[3].plot(r, r**2*xi[:,0], label='vector')

z = 0.466
klin = np.logspace(-3, 0, 2048)
plin = jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
ax[2].plot(klin, plin, label='linear')
ax[2].legend(loc='ll')
r, xi = xicalc_trapz(klin, plin, 2., r)
ax[3].plot(r, r**2*xi, label='linear')
ax[3].legend()
ax[3].format(xlabel='$s$ [Mpc / $h$]', ylabel=r'$s^2\xi$')#, yscale='symlog')
ax[0].format(title='$\delta(x)$')
ax[1].format(title='$\delta(x)$ Pylians')

fig.savefig("plots/jax-cic.png", dpi=300)

exit()
plin = jc.power.linear_matter_power(jc.Planck15(), k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)

def loss(w):
    nbins = 64
    delta = jnp.zeros((n_bins, n_bins, n_bins))
    delta = cic_mas_vec(delta,
                particles[:,0], particles[:,1], particles[:,2], w, 
                n_part, 
                particles[:,0].min(), particles[:,1].min(), particles[:,2].min(),
                box_size,
                n_bins,
                True)
    delta /= delta.mean()
    delta -= 1.
    k, pk, modes = powspec_vec_fundamental(delta, box_size) 
    mask = k < k_ny
    k = k[mask]
    pk = pk[mask, 0]
    plin = jc.power.linear_matter_power(jc.Planck15(), k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
    return jnp.nanmean(pk - plin)**2

grad_loss = jax.grad(loss)
learning_rate = 1e-5
key, subkey = jax.random.split(key)
w = jax.random.normal(key, (particles.shape[0],))* 0.1 + 0.5
for _ in range(100):
    print("Loss", loss(w))
    grad = grad_loss(w)
    print("Grad", grad)
    w -= learning_rate * grad
    print("w mean", w.mean())
    print("w std", w.std())
    





