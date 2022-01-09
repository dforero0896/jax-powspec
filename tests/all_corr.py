import time
import jax
import jax.numpy as jnp
from jax.experimental import loops
import sys
#sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import compute_all_correlations



n_bins = 256
box_size = 1000.
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)




particles = np.loadtxt("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), dtype=np.float32)
#particles = np.load("data/lognormal_nobao_corrected.npy")
mask = ((particles < box_size) & (particles > 0)).all(axis=1)
particles = jax.device_put(particles[mask].copy())
n_part = particles.shape[0]

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

k1, k2 = 0.1, 0.2
theta = jnp.linspace(0, jnp.pi, 20)
s_edges = jnp.arange(1e-3, 200, 5)
k_edges = jnp.arange(5e-3, 1, 2 * jnp.pi / box_size)
s = time.time()
k3D, Pk3D, Nmodes3D_pk, r3D, xi3D, Nmodes3D_xi, k_all, Pk, theta, B, Q = compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta)
print(f"correlations took {time.time() - s} s.", flush=True)

fig, ax = pplt.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
ax[0].plot(k3D, Pk3D[:,0])
ax[1].plot(k3D, Pk3D[:,1])
ax[2].plot(k3D, Pk3D[:,2])
ax[3].plot(r3D, r3D**2*xi3D[:,0])
ax[4].plot(r3D, r3D**2*xi3D[:,1])
ax[5].plot(r3D, r3D**2*xi3D[:,2])
ax[6].plot(theta, B, label='jax-powspec')
ax[7].plot(theta, Q, label='jax-powspec')

ax[0,:].format(xscale='log', yscale='log')
ax.legend()
fig.savefig("plots/all_corr.png", dpi=300)
exit()


import MAS_library as MASL
import Pk_library as PKL
MAS='CIC'
delta = np.zeros((n_bins,n_bins,n_bins), dtype=np.float32)
MASL.MA(np.array(particles).astype(np.float32), delta, box_size, MAS, verbose=True)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
BBk = PKL.Bk(delta, box_size, k1, k2, theta, MAS, 16)
ax[0].plot(theta, BBk.B, label=f'Pylians3', ls='--')
ax[1].plot(theta, BBk.Q, label=f'Pylians3', ls='--')

b1 = 2.2
redshift = 0.466
klin = np.logspace(-3, 0, 2048)
plin = np.array(b1**2 * jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + redshift), transfer_fn=jc.transfer.Eisenstein_Hu))


theta_theory, Bk_theory = PKL.Bispectrum_theory(klin, plin, k1, k2)
print(theta_theory.shape)
ax[0].plot(theta_theory, Bk_theory)

ax[0].format(xlabel=r'$\theta$', ylabel=r'$B(\theta)$', yscale='log')
ax[1].format(xlabel=r'$\theta$', ylabel=r'$Q(\theta)$', yscale='linear')
ax.legend()
fig.savefig("plots/all_corr.png", dpi=300)




