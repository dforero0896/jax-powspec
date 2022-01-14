import numba
import numpy as np
import sys
sys.path.insert(0, '/home/astro/dforero/projects/jax-powspec/')
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")


print("Creating a Log-normal catalog in the CPU", flush=True)
klin = np.logspace(-3, 100, 4056)
redshift = 0.55
b1 = 1.1
seed = 100
box_size = 1000.
use_nbk = False
grid = 256
density = 3.5e-3



if use_nbk:
    from nbodykit.lab import *
    cosmo = cosmology.Planck15
    print(cosmo)
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

    cat = LogNormalCatalog(Plin=Plin, nbar=density, BoxSize=box_size, Nmesh=grid, bias=b1, seed=seed)
    particles = cat['Position'].compute()
    np.save("data/lognormal.npy", particles)
else:
    
    from src.gauss_field import gaussian_field, populate_field
    from src.populate_field import populate_field
    import scipy
    
    #sys.path.insert(0, "/home/astro/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
    import jax_cosmo as jc

    cosmo = jc.Planck15()
    klin = np.linspace(1e-4, 10, 4056)
    #klin = np.logspace(-4, 2, 4056)
    plin_i = np.array(jc.power.linear_matter_power(cosmo, jnp.array(klin), a = 1. / (1 + redshift), transfer_fn=jc.transfer.Eisenstein_Hu))
    Plin = lambda k: jnp.interp(k, klin, plin_i)
    gaussian_field_jit = numba.njit(gaussian_field, fastmath=True)
    gaussian = scipy.fft.irfftn((gaussian_field_jit(grid, klin, plin_i, 0, seed, box_size)), (grid, grid, grid), workers=32)

    lognormal = np.exp(b1 * gaussian)
    lognormal /= lognormal.mean()
    print(lognormal.min(), lognormal.mean(), lognormal.max())
    key = jax.random.PRNGKey(seed+1)
    key, subkey = jax.random.split(key)
    print("Populating field", flush=True)
    particles = populate_field(jnp.array(lognormal), grid, box_size, density, subkey)
    #np.save("data/lognormal.npy", np.array(particles))
    print(particles.min(axis=0), particles.max(axis=0))
    

from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, bispec, compute_all_correlations
import proplot as pplt

particles = jnp.array(particles)
k_ny = jnp.pi * grid / box_size
theta = jnp.linspace(0, jnp.pi, 15)
k1, k2 = 0.05, 0.1
k_edges = jnp.arange(2 * jnp.pi / box_size, k_ny, 2 * jnp.pi / box_size)
s_edges = jnp.arange(1e-3, 200, 5.)
s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])


delta = jnp.zeros((grid, grid, grid))
delta = cic_mas_vec(delta,
            particles[:,0], particles[:,1], particles[:,2], jnp.ones(particles.shape[0]), 
            particles.shape[0], 
            0., 0., 0.,
            box_size,
            grid,
            True)
delta /= delta.mean()
delta -= 1.
print(delta.min(), delta.mean(), delta.max())
k, pk, _, s, xi, _, _,_, _, B, Q = compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta)

fig, ax = pplt.subplots(ncols=3, nrows=3, sharex=False, sharey=False)
ax[0].plot(k, k*(pk[:,0] - 1. / density))
ax[1].plot(k, k*pk[:,1])
ax[2].plot(k, k*pk[:,2])
ax[0].plot(k, k*(b1**2*Plin(k)), ls='--')

ax[3].plot(s, s**2*xi[:,0])
ax[4].plot(s, s**2*xi[:,1])
ax[5].plot(s, s**2*xi[:,2])

ax[6].imshow(delta.mean(axis=0), colorbar='right')
ax[7].plot(theta, B)
ax[8].plot(theta, Q)

fig.savefig("plots/create_lognormal.png", dpi=300)




