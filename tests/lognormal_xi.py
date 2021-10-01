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
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, xi_vec_coords

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
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    
    cat = LogNormalCatalog(Plin=Plin, nbar=3.5e-3, BoxSize=box_size, Nmesh=256, bias=b1, seed=seed)
    particles = cat['Position'].compute()
    np.save("data/lognormal.npy", particles)
else:

    particles = np.load("data/lognormal.npy")

particles = jax.device_put(particles)
cosmo = jc.Planck15()






key = jax.random.PRNGKey(5)

n_part = particles.shape[0]
key, subkey = jax.random.split(key)
w = jnp.ones(particles.shape[0])
print(n_part)


shot_noise = box_size**3 / n_part

z = redshift
n_bins = 128
klin = jnp.logspace(-3, 0, 2048)
sarr = jnp.linspace(5, 195, 1000)
plin = b1**2 * jc.power.linear_matter_power(cosmo, klin, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) + shot_noise
sarr, xi0, xi2, xi4 = xicalc_trapz(klin, plin, 2., sarr)
k_edges = jnp.arange(0.005, 0.2, 0.003)
s_coords = xi_vec_coords(n_bins, box_size, k_edges)

target_0 = jnp.interp(s_coords, sarr, xi0)
target_2 = jnp.interp(s_coords, sarr, xi2)
target_4 = jnp.interp(s_coords, sarr, xi4)



#@jax.jit
def loss(positions):
    xpos = positions[:,0]
    ypos = positions[:,1]
    zpos = positions[:,2]

    k_ny = jnp.pi * n_bins / box_size
    
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
    s, xi, modes = xi_vec(delta, box_size, k_edges) 
    print((xi[:,0]))
    print(target_0)
    return jnp.nanmean((s**2*xi[:,0] - s**2*target_0)**2)


from jax.experimental import optimizers
key, subkey = jax.random.split(key)
#w0 = jax.random.normal(key, (particles.shape[0],))* 0.2 + 0.4

x0 = particles
learning_rate = 1e-5
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(x0)

@jax.jit
def step(step, opt_state):
    value, grads = jax.value_and_grad(loss)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 10
s = time.time()
print("Training...", flush=True)
#opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
for i in range(num_steps):
    value, grads = jax.value_and_grad(loss)(get_params(opt_state))
    opt_state = opt_update(i, grads, opt_state)
    print(value)
    #print(grads)
print(f"Training took {time.time() - s} s", flush=True)
new_pos = get_params(opt_state)
exit()
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
s, xi0, xi2, xi4 = xicalc_trapz(klin, plin, 2., s)
ax[3].plot(s, s**2*xi0, label='linear', ls=':')
ax[4].plot(s, s**2*xi2, label='linear', ls=':')
ax[5].plot(s, s**2*xi4, label='linear', ls=':')


ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[1].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[1].format(title='Log-normal')
ax[0].legend(loc='bottom')
ax[3].legend(loc='bottom')
ax[3].format(xlabel='$s$', ylabel=r'$s^2\xi$')
fig.savefig("plots/lognormal_xi.png", dpi=300)









