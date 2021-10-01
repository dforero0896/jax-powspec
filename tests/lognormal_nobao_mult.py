from nbodykit.lab import *
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz, xicalc_trapz_linear
import numpy as np
import pandas as pd
import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, kaiser_power_spectrum, xi_vec_coords

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






key = jax.random.PRNGKey(5)

n_part = particles.shape[0]
key, subkey = jax.random.split(key)
w = jnp.ones(particles.shape[0])
print(n_part)


shot_noise = box_size**3 / n_part

z = redshift
n_bins = 256
k_ny = jnp.pi * n_bins / box_size
growth_rate = jc.background.growth_rate(cosmo, jnp.array([1. / (1. + z)])).squeeze()
k_edges = jnp.arange(0.003, k_ny, 0.0025)
k = 0.5 * (k_edges[1:] + k_edges[:-1])
s_edges = jnp.arange(0.005, 0.35, 0.005)
plin =  jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
plin_nw =  jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu, type='eisenhu')
pk0, pk2, pk4 = kaiser_power_spectrum(plin, b1, growth_rate)
pk0 += shot_noise
pk0_nw, pk2_nw, pk4_nw = kaiser_power_spectrum(plin_nw, b1, growth_rate)
pk0_nw += shot_noise
s = xi_vec_coords(n_bins, box_size, s_edges)
s, xi0, _, _ = xicalc_trapz_linear(k, pk0, 2., s)
s, _, xi2, _ = xicalc_trapz_linear(k, pk2, 2., s)
s, _, _, xi4 = xicalc_trapz_linear(k, pk4, 2., s)

s, xi0_nw, _, _ = xicalc_trapz_linear(k, pk0_nw, 2., s)
s, _, xi2_nw, _ = xicalc_trapz_linear(k, pk2_nw, 2., s)
s, _, _, xi4_nw = xicalc_trapz_linear(k, pk4_nw, 2., s)

target_pk0 = pk0 / pk0_nw - 1.
target_pk2 = pk2 / pk2_nw - 1.
target_pk4 = pk4 / pk4_nw - 1.


target_xi0 = s**2 * (xi0 - xi0_nw) #- 1.
target_xi2 = s**2 * (xi2 - xi2_nw) #- 1.
target_xi4 = s**2 * (xi4 - xi4_nw) #- 1.
fig, ax = pplt.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
ax[3].plot(s, s**2*xi0, label='linear', ls=':')
ax[4].plot(s, s**2*xi2, label='linear', ls=':')
ax[5].plot(s, s**2*xi4, label='linear', ls=':')

ax[3].plot(s, s**2*xi0_nw, label='linear nw', ls='-.')
ax[4].plot(s, s**2*xi2_nw, label='linear nw', ls='-.')
ax[5].plot(s, s**2*xi4_nw, label='linear nw', ls='-.')

#ax[0].plot(s, target_xi0, label='Linear nw', ls='-.')
#ax[1].plot(s, target_xi2, ls='-.')
#ax[2].plot(s, target_xi4, ls='-.')

fig.savefig("plots/lognormal_nobao_multi.png", dpi=300)



@jax.jit
def loss(positions):
    xpos = positions[:,0]
    ypos = positions[:,1]
    zpos = positions[:,2]
      

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
    k, pk, _ = powspec_vec(delta, box_size, k_edges) 
    s_centers, xi, modes = xi_vec(delta, box_size, s_edges) 

    #return jnp.nanmean((jnp.log10(pk[:,0]) - jnp.log10(pk0))**2) #+ jnp.nanmean(jnp.abs(pk[:,1])) * 1e-2# + jnp.nanmean(jnp.abs(pk[:,1])) * 1e-4
    #loss_mono = jnp.nanmean(jnp.abs(pk[:,0] - pk0))
    #loss_quad = jnp.nanmean(jnp.abs(pk[:,1] - pk2)) 
    #loss_hexa = jnp.nanmean(jnp.abs(pk[:,2] - pk4))
    loss_mono = jnp.nanmean(((pk[:,0] - pk0))**2) #+ jnp.nanmean((xi[:,0] - xi0)**2)
    loss_quad = jnp.nanmean(((pk[:,1] - pk2))**2) 
    loss_hexa = jnp.nanmean(((pk[:,2] - pk4))**2) 

    #loss_mono = jnp.nanmean((((pk[:,0]/pk0_nw - 1)) - target_pk0)**2) #+ jnp.nanmean(((s**2 * (xi[:,0] - xi0_nw)) - target_xi0)**2)
    #loss_quad = jnp.nanmean((((pk[:,1]/pk2_nw - 1)) - target_pk2)**2) 
    #loss_hexa = jnp.nanmean((((pk[:,2]/pk4_nw - 1)) - target_pk4)**2) 

    return loss_mono + loss_quad + loss_hexa 

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
num_steps = 800
s = time.time()
print("Training...", flush=True)
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
print(f"Training took {time.time() - s} s", flush=True)
new_pos = get_params(opt_state)
#final_loss = loss(w).block_until_ready()
#print(f"Final loss: {final_loss}", flush=True)
#print(particles)
#print(new_pos)


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
pk = pk[mask]
s, xi, modes = xi_vec_fundamental(delta, box_size)
mask = s < 200
s = s[mask]
xi = xi[mask]

bias = b1
plin =  jc.power.linear_matter_power(cosmo, k, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) 
pk0, pk2, pk4 = kaiser_power_spectrum(plin, bias, growth_rate)
pk0 += shot_noise
plin_nw =  jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu, type='eisenhu')
pk0_nw, pk2_nw, pk4_nw = kaiser_power_spectrum(plin_nw, b1, growth_rate)
pk0_nw += shot_noise
ax[-2].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[-2].format(title='Corrected')
ax[0].plot(k, k*pk[:, 0], label='Corrected mono')
ax[1].plot(k, k*pk[:, 1], label='')
ax[2].plot(k, k*pk[:, 2], label='')
ax[0].plot(k, k*pk0, label='Linear', ls=':')
ax[1].plot(k, k*pk2, ls=':')
ax[2].plot(k, k*pk4, ls=':')
ax[0].plot(k, k*pk0_nw, label='Linear nw', ls='-.')
ax[1].plot(k, k*pk2_nw, ls='-.')
ax[2].plot(k, k*pk4_nw, ls='-.')
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
pk = pk[mask]
#s, xi, modes = xi_vec_fundamental(delta, box_size)
s, xi, modes = xi_vec(delta, box_size, s_edges)
mask = s < 200
s = s[mask]
xi = xi[mask]

ax[0].plot(k, k*pk[:, 0], label='Log-normal', ls='--')
ax[1].plot(k, k*pk[:, 1], label='Corrected quad')
ax[2].plot(k, k*pk[:, 2], label='Corrected hexa')
ax[3].plot(s, s**2*xi[:,0], label='Log-normal', ls = '--')
ax[4].plot(s, s**2*xi[:,1], label='Log-normal')
ax[5].plot(s, s**2*xi[:,2], label='Log-normal')


ax[0,:].format(xscale = 'log', yscale = 'linear', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[-1].imshow(delta[:,:,:].mean(axis=2), colorbar='r')
ax[-1].format(title='Log-normal')
ax[0].legend(loc='bottom')
ax[3].legend(loc='bottom')
ax[3].format(xlabel='$s$', ylabel=r'$s^2\xi$')
fig.savefig("plots/lognormal_nobao_multi.png", dpi=300)

print("Saving corrected catalog...", flush=True)
np.save("data/lognormal_nobao_multi_corrected.npy", (new_pos + box_size) % box_size)








