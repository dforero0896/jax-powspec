import numpy as np
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz, xicalc_trapz_linear
import jax.numpy as jnp
import proplot as pplt
from src.correlations import s_edges_conv, xi_vec_coords

cosmo = jc.Planck15()
z = 0.55
b1 = 2.
box_size = 1000.
n_bins = 256
n_part = 100000
k_ny = jnp.pi * n_bins / box_size
#growth_rate = jc.background.growth_rate(cosmo, jnp.array([1. / (1. + z)])).squeeze()
s_edges = jnp.arange(0.005, 0.35, 0.005)
k_edges = jnp.arange(0.003, k_ny, 0.0025)
k = 0.5 * (k_edges[1:] + k_edges[:-1])
dk = k[1] - k[0]
Pk = b1**2 * jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
shot_noise = box_size**3 / n_part

cov = (2*jnp.pi)**3 / box_size**3 * (2 * (Pk + shot_noise)**2 / (4*jnp.pi*k**2 * dk))

fig, ax = pplt.subplots(nrows=2, ncols=2, sharey=False, sharex=False)

ax[0].errorbar(k, Pk, yerr = jnp.sqrt(cov))
ax[1].plot(k, cov)

ax[0,:].format(xscale='log', yscale='log', xformatter='log', yformatter='log')

fig.savefig("plots/estimate_pk_covariance.png", dpi=300)

a_damp=3.
s_edges = jnp.linspace(0, 200, 41)
s = 0.5 * (s_edges[1:] + s_edges[:-1])
s, xi0, _, _ = xicalc_trapz_linear(k, Pk, a_damp, s)

ks = k[:,None] * s[None,:]
damp = jnp.exp(-k**2*a_damp**2)
j0 = jnp.sin(ks) / ks
j0sq = jnp.einsum('ia,ib->iab', j0, j0)
print(j0sq.shape)
factor = dk**2 * k**4 * cov * damp**2 / (2*jnp.pi**2)**2
cov_conf = (factor[:,None,None] * j0sq).sum(axis=0)
var_conf = jnp.diag(cov_conf)
cov_mat = jnp.diag(cov)


ax[3].imshow(cov_conf / jnp.sqrt(var_conf[:,None] * var_conf[None,:]))

ax[2].errorbar(s, s**2*xi0, yerr = s**2*jnp.sqrt(jnp.diag(cov_conf)))

fig.savefig("plots/estimate_pk_covariance.png", dpi=300)

