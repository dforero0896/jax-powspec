import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import numpy as np
import pandas as pd
import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental

import MAS_library as MASL
import Pk_library as PKL



box_size = 1000.

key = jax.random.PRNGKey(5)

n_part = 3000000
key, subkey = jax.random.split(key)
particles = box_size * jax.random.uniform(subkey, (n_part,3))
w = jnp.ones(particles.shape[0])
print(n_part)


shot_noise = box_size**3 / n_part

z = 0.466
n_bins = 256

@jax.jit
def loss(positions):
    xpos = positions[:,0]
    ypos = positions[:,1]
    zpos = positions[:,2]
    bias = 1.#2.2
    
    
    k_ny = jnp.pi * n_bins / box_size
    k_edges = jnp.arange(0.003, k_ny, 0.0025)
    delta = jnp.zeros((n_bins, n_bins, n_bins))
    delta = cic_mas_vec(delta,
                xpos, ypos, zpos, w, 
                n_part, 
                0., 0., 0.,
                box_size,
                n_bins,
                True)
    delta /= delta.mean()
    delta -= 1.
    k, pk, modes = powspec_vec(delta, box_size, k_edges) 
    pk = pk[:,0] 
    plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
    return jnp.nanmean((jnp.log10(pk) - jnp.log10(plin))**2)

from jax.experimental import optimizers
key, subkey = jax.random.split(key)
#w0 = jax.random.normal(key, (particles.shape[0],))* 0.2 + 0.4

x0 = particles
learning_rate = 1.
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(x0)
@jax.jit
def step(step, opt_state):
    value, grads = jax.value_and_grad(loss)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 2000
s = time.time()
print("Training...", flush=True)
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
print(f"Training took {time.time() - s} s", flush=True)
new_pos = get_params(opt_state)
#final_loss = loss(w).block_until_ready()
#print(f"Final loss: {final_loss}", flush=True)
print(particles)
print(new_pos)


k_ny = jnp.pi * n_bins / box_size
delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
            new_pos[:,0], new_pos[:,1], new_pos[:,2], w, 
            n_part, 
            0., 0., 0.,
            box_size,
            n_bins,
            True)
delta /= delta.mean()
delta -= 1.

k, pk, modes = powspec_vec_fundamental(delta, box_size) 
mask = k < k_ny
k = k[mask]
pk = pk[mask, 0]
s, xi, modes = xi_vec_fundamental(delta, box_size)
mask = s < 200
s = s[mask]
xi = xi[mask,0]

bias = 1.#2.2
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)

fig, ax = pplt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
ax[2].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[2].format(title='Corrected')
ax[0].plot(k, pk, label='Corrected')
ax[0].plot(k, plin, label='linear', ls=':')
ax[3].plot(s, s**2*xi, label='Corrected')



delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
            particles[:,0], particles[:,1], particles[:,2], w, 
            n_part, 
            0., 0., 0.,
            box_size,
            n_bins,
            True)
delta /= delta.mean()
delta -= 1.
k, pk, modes = powspec_vec_fundamental(delta, box_size) 
mask = k < k_ny
k = k[mask]
pk = pk[mask, 0]
s, xi, modes = xi_vec_fundamental(delta, box_size)
mask = s < 200
s = s[mask]
xi = xi[mask,0]

ax[0].plot(k, pk, label='Raw', ls='--')
ax[3].plot(s, s**2*xi, label='Raw', ls = '--')
klin = np.logspace(-3, 0, 2048)
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
s, xi = xicalc_trapz(klin, plin, 2., s)
ax[3].plot(s, s**2*xi, label='linear', ls=':')

ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[1].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[1].format(title='Raw')
ax[0].legend(loc='bottom')
ax[3].legend(loc='bottom')
ax[3].format(xlabel='$s$', ylabel=r'$s^2\xi$')
fig.savefig("plots/mocks.png", dpi=300)



