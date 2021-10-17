import numpy as np
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz, xicalc_trapz_linear
import jax.numpy as jnp
import proplot as pplt
from src.correlations import s_edges_conv, xi_vec_coords, xi_vec, powspec_vec, estimate_pk_variance, estimate_xi_covariance
from src.mas import cic_mas_vec
import pandas as pd
import jax

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

a_damp=2.
s_edges = jnp.linspace(0, 200, 41)
#s_edges = s_edges[(s_edges >= 50.) & (s_edges <= 150.)]
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


#ax[3].imshow(cov_conf / jnp.sqrt(var_conf[:,None] * var_conf[None,:]))
ax[3].plot(s, s**2*jnp.diag(cov_conf))

ax[2].errorbar(s, s**2*xi0, yerr = s**2*jnp.sqrt(jnp.diag(cov_conf)))

fig.savefig("plots/estimate_pk_covariance.png", dpi=300)

#exit()
box_size = 1000.
voids = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.VOID.dat", usecols = (0,1,2,3), delim_whitespace=True, engine='c').values.astype(np.float32)
mask = ((voids[:,:3] < box_size) & (voids[:,:3] > 0)).all(axis=1) 
voids = jax.device_put(voids[mask].copy())
n_voids = voids.shape[0]
w = jax.nn.sigmoid(10 * (voids[:,3] - 16.))

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
s_edges = jnp.linspace(0, 200, 41)
#s_edges = s_edges[(s_edges >= 50.) & (s_edges <= 150.)]
s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
shot_noise = box_size**3 / n_voids #* w.sum()**2 / ((w**2).sum())

k_edges = s_edges_conv(n_bins, box_size, s_edges)
s_centers, xiv, modes = xi_vec(delta_v, box_size, k_edges)


k_edges = jnp.arange(1e-2, k_ny, 0.005)
kp, pk, modes = powspec_vec(delta_v, box_size, k_edges) 
dk = kp[1] - kp[0]
pk_var = estimate_pk_variance(kp, pk[:,0], box_size, 0, dk)


xi_cov = estimate_xi_covariance(s_centers, kp, pk, a_damp, pk_var, dk)

ax[0].errorbar(kp, pk[:,0], jnp.sqrt(pk_var))
ax[1].plot(kp, jnp.sqrt(pk_var))

ax[2].errorbar(s_centers, s_centers**2*xiv[:,0], s_centers**2*jnp.sqrt(jnp.diag(xi_cov)))
ax[3].plot(s_centers, s_centers**2*jnp.sqrt(jnp.diag(xi_cov)))
ax[3].format(yscale='log')







fig.savefig("plots/estimate_pk_covariance.png", dpi=300)