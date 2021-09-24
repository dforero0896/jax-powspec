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



box_size = 2000.

key = jax.random.PRNGKey(42)

particles = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), delim_whitespace=True, engine='c').values.astype(np.float32)
voids = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.VOID.dat", usecols = (0,1,2,3), delim_whitespace=True, engine='c').values.astype(np.float32)
mask = ((particles < box_size) & (particles > 0)).all(axis=1)
particles = jax.device_put(particles[mask].copy())
n_part = particles.shape[0]
mask = ((voids[:,:3] < box_size) & (voids[:,:3] > 0)).all(axis=1) & (voids[:,3] > 16.)
voids = jax.device_put(voids[mask].copy())
n_voids = voids.shape[0]
w = jnp.ones(voids.shape[0])
w0 = jnp.ones(particles.shape[0])
print("Number of galaxies", n_part)
print("Number of voids", n_voids)


shot_noise = box_size**3 / n_part
shot_noise_v = box_size**3 / n_voids
z = 0.466
n_bins = 256
delta_v = jnp.zeros((n_bins, n_bins, n_bins))
delta_v = cic_mas_vec(delta_v,
            voids[:,0], voids[:,1], voids[:,2], w, 
            n_voids, 
            0., 0., 0.,
            box_size,
            n_bins,
            True)
delta_v /= delta_v.mean()
delta_v -= 1.
k_ny = jnp.pi * n_bins / box_size
k_edges = jnp.arange(0.003, k_ny, 0.0025)
k, pk, modes = powspec_vec(delta_v, box_size, k_edges) 
pkv = pk[:, 0] - shot_noise_v

@jax.jit
def loss(positions):
    xpos = positions[:,0]
    ypos = positions[:,1]
    zpos = positions[:,2]

    
    delta = jnp.zeros((n_bins, n_bins, n_bins))
    delta = cic_mas_vec(delta,
                xpos, ypos, zpos, w0, 
                n_part, 
                0., 0., 0.,
                box_size,
                n_bins,
                True)
    delta /= delta.mean()
    delta -= 1.

    k, pk, modes = powspec_vec(delta, box_size, k_edges) 
    pk = pk[:,0] - shot_noise
    
    return jnp.nanmean((jnp.log10(pk) - jnp.log10(pkv))**2)

from jax.experimental import optimizers
key, subkey = jax.random.split(key)
#w0 = jax.random.normal(key, (particles.shape[0],))* 0.2 + 0.4

x0 = particles
learning_rate = 1
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(x0)
@jax.jit
def step(step, opt_state):
    value, grads = jax.value_and_grad(loss)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 1000
s = time.time()
print("Training...", flush=True)
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
print(f"Training took {time.time() - s} s", flush=True)
new_pos = get_params(opt_state)
#final_loss = loss(w).block_until_ready()
#print(f"Final loss: {final_loss}", flush=True)
print(particles)
print(new_pos)



delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
            new_pos[:,0], new_pos[:,1], new_pos[:,2], w0, 
            n_voids, 
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
bias = 2.2
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)

fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
ax[2].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[2].format(title='Corrected')
ax[0].plot(k, pk, label='Corrected')
ax[0].plot(k, plin, label='linear', ls=':')
s, xi, modes = xi_vec_fundamental(delta, box_size)
mask = s < 200
ax[4].plot(s[mask], s[mask]**2*xi[mask,0], label='Corrected')


delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
            particles[:,0], particles[:,1], particles[:,2], w0, 
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
ax[0].plot(k, pk, label='Raw', ls='--')
ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[1].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[1].format(title='Raw')



k, pk, modes = powspec_vec_fundamental(delta_v, box_size) 
mask = k < k_ny
k = k[mask]
pk = pk[mask, 0]
ax[0].plot(k, pk, label='Void', ls='--')
ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[3].imshow(delta_v[:,:,:].mean(axis=2), colorbar='r')
ax[3].format(title='Void')
ax[0].legend(loc='bottom')

s, xi, modes = xi_vec_fundamental(delta_v, box_size)
mask = s < 200
ax[4].plot(s[mask], s[mask]**2*xi[mask,0], label='Voids')
ax[4].legend(loc='bottom')
ax[4].format(xlabel = '$s$', ylabel=r'$s^2\xi$')








fig.savefig("plots/voids.png", dpi=300)