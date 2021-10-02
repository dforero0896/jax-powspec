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
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, xi_vec_coords, s_edges_conv

nsbin = 1000
inv_cov = jnp.array(np.load("data/inv_cov_void.npy"))
cov = jnp.array(np.load("data/cov_void.npy"))
sarr = jnp.linspace(1e-3, 200, nsbin)
smooth_a = 2.
kmin=2.5e-3
kmax = 2.
lnk = np.linspace(np.log(kmin), np.log(kmax), 2048)
k = np.exp(lnk)
k = np.logspace(-3, 1, 2048)
cosmo = jc.Planck15()
z = 0.466
box_size = 1000.
gain = 10.
cut_position = 18.

plin_nw = jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu, type='eisenhu')
plin = jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)

@jax.jit
def template_pk(Sigma_nl, c, ksq, plin, plin_nw):
    return ((plin - plin_nw) * jnp.exp(- 0.5 * ksq * Sigma_nl**2) + plin_nw) * (1 + c*k**2)
@jax.jit
def template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a):
    pk_temp = template_pk(Sigma_nl, c, k**2, plin, plin_nw)
    sarr, xi, xi2, xi4 = xicalc_trapz(k, pk_temp, smooth_a, sarr)
    return xi

def model_curve(params, s):
    alpha, Sigma_nl, B, c, a1, a2, a3 = params
    xi_model = jnp.interp(s * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    return (B**2 * xi_model + a1 / s**2 + a2 / s + a3)
def chisq(params, s_obs, xi_obs):
    alpha, Sigma_nl, B, c, a1, a2, a3 = params
    xi_model = jnp.interp(s_obs * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    e =  (B**2 * xi_model + a1 / s_obs**2 + a2 / s_obs + a3) - xi_obs
    return 0.5 * e.T.dot(e) / e.shape[0]
def neg_loglike(params, s_obs, xi_obs):
    return chisq(params, s_obs, xi_obs)

neg_F = jax.hessian(neg_loglike)
score_func = jax.grad(neg_loglike)



voids = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.VOID.dat", usecols = (0,1,2,3), delim_whitespace=True, engine='c').values.astype(np.float32)
mask = ((voids[:,:3] < box_size) & (voids[:,:3] > 0)).all(axis=1) 
voids = jax.device_put(voids[mask].copy())
n_voids = voids.shape[0]

w = jax.nn.sigmoid(gain * (voids[:,3] - cut_position))


n_bins = 300


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
s_edges = jnp.arange(0, 205, 5)
k_edges = s_edges_conv(n_bins, box_size, s_edges)
s_centers, xiv, modes = xi_vec(delta_v, box_size, k_edges) 
params = (1.05, 9., 1.73, 90., -3.5, -3.8e-2, 0.)
params_damp = (1.05, 25., 1.73, 90., -3.5, -3.8e-2, 0.)

xi_model = model_curve(params, s_centers)
fisher = jnp.array(neg_F(params, s_centers, xiv[:,0]))
#print("Initial Fisher information: ", jnp.diag(fisher))
#print("Initial scores: ", - jnp.array(score_func(params, s_centers, xiv[:,0])))
fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
ax[0].plot(s_centers, s_centers**2*xiv[:,0])
ax[0].plot(s_centers, s_centers**2*xi_model)

xi_model = model_curve(params_damp, s_centers)
ax[0].plot(s_centers, s_centers**2*xi_model)
ax[1].plot(k, plin_nw)
ax[1].plot(k, plin)
ax[1].format(xscale='log', yscale='log')
fig.savefig("plots/void-model.png", dpi=300)


param_cov = np.linalg.inv(fisher)
print("Final parameter variances: ", jnp.sqrt(jnp.diag(param_cov)))


pk_temp = template_pk(params[1], params[2], k**2, plin, plin_nw)
k_edges = jnp.arange(1e-2, k_ny, 0.001)
kp, pk, modes = powspec_vec(delta_v, box_size, k_edges) 

ax[1].plot(k, pk_temp, label='template')
ax[1].plot(kp, pk[:,0], label='measured')
ax[1].legend(loc='bottom')

fig.savefig("plots/void-model.png", dpi=300)