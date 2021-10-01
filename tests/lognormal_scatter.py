from nbodykit.lab import *
import numpy as np
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
redshift = 0.55
b1 = 2.0
seed = 5
create_data = False
if create_data:
    
    cosmo = cosmology.Planck15
    print(cosmo)
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='NoWiggleEisensteinHu')
    
    cat = LogNormalCatalog(Plin=Plin, nbar=3.5e-3, BoxSize=box_size, Nmesh=450, bias=b1, seed=seed)
    particles = cat['Position'].compute()
    np.save("data/lognormal_nobao.npy", particles)
else:

    particles = np.load("data/lognormal_nobao.npy")

particles = jax.device_put(particles)
cosmo = jc.Planck15()






key = jax.random.PRNGKey(seed)

n_part = particles.shape[0]
key, subkey = jax.random.split(key)
w = jnp.ones(particles.shape[0])
print(n_part)


shot_noise = box_size**3 / n_part

z = redshift
n_bins = 256

@jax.jit
def loss(params, subkey):
    positions, sigma_scatter = params
    xpos = positions[:,0]
    ypos = positions[:,1]
    zpos = positions[:,2]
    bias = b1#2.2
    scatter = jax.random.normal(subkey, positions.shape) * sigma_scatter
    k_ny = jnp.pi * n_bins / box_size
    k_edges = jnp.arange(0.003, k_ny, 0.0025)
    delta = jnp.zeros((n_bins, n_bins, n_bins))
    delta = cic_mas_vec(delta,
                #xpos + scatter[:,0], ypos + scatter[:,1], zpos + scatter[:,2], w, 
                xpos , ypos, zpos, w, 
                n_part, 
                0., 0., 0.,
                box_size,
                n_bins,
                True)
    delta /= delta.mean()
    delta -= 1.
    k, pk, modes = powspec_vec(delta, box_size, k_edges) 
    plin = bias**2 * jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) + shot_noise
    #return jnp.nanmean((jnp.log10(pk[:,0]) - jnp.log10(plin))**2) 
    error = (pk[:,0] - plin)[:-1]
    variances_est = jnp.diff(pk[:,0])**2
    #return jnp.nanmean((pk[:,0] - plin)**2) 
    return error.dot(jnp.diag(1. / variances_est).dot(error))

from jax.experimental import optimizers
key, subkey = jax.random.split(key)
#w0 = jax.random.normal(key, (particles.shape[0],))* 0.2 + 0.4

x0 = particles
learning_rate = 1.
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init((x0, 1e-3))
update_state = dict(opt_state=opt_state, key=key, subkey=subkey)
@jax.jit
def step(step, update_state):
    opt_state = update_state['opt_state']
    subkey = update_state['subkey']
    key = update_state['key']
    value, grads = jax.value_and_grad(loss)(get_params(opt_state), subkey)
    opt_state = opt_update(step, grads, opt_state)
    key, subkey = jax.random.split(key)
    return dict(opt_state=opt_state, key=key, subkey=subkey)
num_steps = 200
s = time.time()
print("Training...", flush=True)
final_state = jax.lax.fori_loop(0, num_steps, step, update_state)   
print(f"Training took {time.time() - s} s", flush=True)
opt_state = final_state['opt_state']
key = final_state['key']
subkey = final_state['subkey']
new_pos, sigma_scatter = get_params(opt_state)
#final_loss = loss(w).block_until_ready()
#print(f"Final loss: {final_loss}", flush=True)
print(particles)
print(new_pos)
print(sigma_scatter)
scatter = jax.random.normal(subkey, new_pos.shape) * sigma_scatter
print(scatter)
new_pos += scatter
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
xi = xi[mask]

bias = b1
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) + shot_noise

fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
ax[2].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[2].format(title='Corrected')
ax[0].plot(k, pk, label='Corrected')
ax[0].plot(k, plin, label='Linear', ls=':')
ax[3].plot(s, s**2*xi[:,0], label='Corrected')
ax[4].plot(s, s**2*xi[:,1], label='Corrected')
ax[5].plot(s, s**2*xi[:,2], label='Corrected')



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
xi = xi[mask]

ax[0].plot(k, pk, label='Log-normal', ls='--')
ax[3].plot(s, s**2*xi[:,0], label='Log-normal', ls = '--')
ax[4].plot(s, s**2*xi[:,1], label='Log-normal')
ax[5].plot(s, s**2*xi[:,2], label='Log-normal')
klin = np.logspace(-3, 0, 2048)
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) + shot_noise
s, xi = xicalc_trapz(klin, plin, 2., s)
ax[3].plot(s, s**2*xi, label='linear', ls=':')


ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[1].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[1].format(title='Log-normal')
ax[0].legend(loc='bottom')
ax[3].legend(loc='bottom')
ax[3].format(xlabel='$s$', ylabel=r'$s^2\xi$')
fig.savefig("plots/lognormal_scatter.png", dpi=300)

#print("Saving corrected catalog...", flush=True)
#np.save("data/lognormal_scatter_corrected.npy", (new_pos + box_size) % box_size)








