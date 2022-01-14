
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import sys
sys.path.insert(0, "/home/astro/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import numpy as np

import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import compute_all_correlations, compute_2pt_correlations, bispec, powspec_vec, powspec_vec_fundamental

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



k1, k2 = 0.1, 0.2
theta = jnp.linspace(0, jnp.pi, 20)
s_edges = jnp.arange(1e-3, 200, 5)
s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
k_edges = jnp.arange(5e-3, 1, 2 * jnp.pi / box_size)



key = jax.random.PRNGKey(5)

n_part = particles.shape[0]
key, subkey = jax.random.split(key)
w = jnp.broadcast_to(jnp.array([1.]), (particles.shape[0],))

shot_noise = box_size**3 / n_part

z = redshift
n_bins = 256
klin = np.logspace(-3, 0, 2048)
plin_i = b1**2 * jc.power.linear_matter_power(cosmo, klin, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) 
_, xi0, xi2, xi4 = xicalc_trapz(klin, plin_i, 2., s_centers)
theta_theory, Bk_i = PKL.Bispectrum_theory(klin, plin_i, k1, k2)
theta = jnp.linspace(0, jnp.pi, 15)




@jax.jit
def loss(positions):

    #positions += box_size
    #positions %= box_size
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
    
    #k, pk, _, s, xi, _, _,_, _, B, Q = compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta)
    _, _, _, B, Q = bispec(delta, box_size, k1, k2, theta)
    k, pk, _ = powspec_vec(delta, box_size, k_edges) 
        
    plin = b1**2 * jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)  + shot_noise
    
    #mono_loss = jnp.nanmean((jnp.log10(pk[:,0]) - jnp.log10(plin))**2)
    mono_loss_k = jnp.nanmean((pk[:,0] - plin)**2)
    #mono_loss_s = jnp.nanmean((xi[:,0] - xi0)**2)
    

    bispec_loss = 1e-9 * jnp.nanmean((B - jnp.interp(theta, theta_theory, Bk_i))**2)
    
    #return mono_loss #+ 0.01 * quad_loss + 0.01 * hexa_loss
    return mono_loss_k + bispec_loss 



from jax.experimental import optimizers



# Setup training loop


x0 = particles
learning_rate = 1.
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(x0)
@jax.jit
def step(step, opt_state):
    grads = jax.grad(loss)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 100
s = time.time()
print("Training...", flush=True)
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
print(f"Training took {time.time() - s} s", flush=True)
new_pos = get_params(opt_state)
new_pos = (new_pos + box_size) % box_size

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

k, pk, _, s, xi, _, _,_, _, B, Q = compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta)
k, pk, modes = powspec_vec_fundamental(delta, box_size) 
mask = k < k_ny
k = k[mask]
pk = pk[mask]
s, xi, modes = xi_vec_fundamental(delta, box_size)
mask = s < 200
s = s[mask]
xi = xi[mask]

fig, ax = pplt.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
ax[6].imshow(delta[:,:,:].mean(axis=2), colorbar='r', vmin=-1, vmax = 1.6)
ax[6].format(title='Corrected')
ax[0].plot(k, k*pk[:,0], label='Corrected')
ax[1].plot(k, k*pk[:,1], label='Corrected')
ax[2].plot(k, k*pk[:,2], label='Corrected')

ax[3].plot(s, s**2*xi[:,0], label='Corrected')
ax[4].plot(s, s**2*xi[:,1], label='Corrected')
ax[5].plot(s, s**2*xi[:,2], label='Corrected')

ax[8].plot(theta, B, label='Corrected')

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
k, pk, _, s, xi, _, _,_, _, B, Q = compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta)

ax[0].plot(k, k*pk[:,0], label='Log-normal', ls='--')
ax[1].plot(k, k*pk[:,1], label='Log-normal', ls='--')
ax[2].plot(k, k*pk[:,2], label='Log-normal', ls='--')
ax[3].plot(s, s**2*xi[:,0], label='Log-normal', ls = '--')
ax[4].plot(s, s**2*xi[:,1], label='Log-normal')
ax[5].plot(s, s**2*xi[:,2], label='Log-normal')

ax[8].plot(theta, B, label='Log-normal')

ax[3].plot(s, s**2*xi0, label='linear', ls=':')
ax[4].plot(s, s**2*xi2, label='linear', ls=':')
ax[5].plot(s, s**2*xi4, label='linear', ls=':')




ax[0].plot(klin, klin * plin_i, label='Linear', ls=':')
ax[8].plot(theta_theory, Bk_i, label='Linear', ls=':')

ax[0].format(xscale = 'log', yscale = 'log', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[0,:].format(xscale = 'log', yscale='linear', xlabel='$k$ [$h$/Mpc]', ylabel='$P(k)$')
ax[7].imshow(delta[:,:,:].mean(axis=2), colorbar='r', vmin=-1, vmax = 1.6)
ax[7].format(title='Log-normal')
ax[0].legend(loc='bottom')
ax[3].legend(loc='bottom')
ax[8].legend(loc='bottom')
ax[3].format(xlabel='$s$', ylabel=r'$s^2\xi$')
fig.savefig("plots/lognormal_bispec.png", dpi=300)









