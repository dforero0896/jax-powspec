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
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, xi_vec_coords, s_edges_conv, estimate_xi_covariance, estimate_pk_variance

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
cut_position = 14.
alpha_lambda = 1e-2

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


voids = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.VOID.dat", usecols = (0,1,2,3), delim_whitespace=True, engine='c').values.astype(np.float32)
mask = ((voids[:,:3] < box_size) & (voids[:,:3] > 0)).all(axis=1) & (voids[:,3]>10.)
voids = jax.device_put(voids[mask].copy())
n_voids = voids.shape[0]
inv_cov = jnp.array(np.load("data/inv_cov_void.npy"))
cov = jnp.array(np.load("data/cov_void.npy"))

w = jax.nn.sigmoid(gain * (voids[:,3] - cut_position))


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
s_edges = jnp.linspace(0, 200, 41)
s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
mask_s = (s_centers > 50.) & (s_centers < 150.)
s_edges = s_edges[(s_edges >= 50.) & (s_edges <= 150.)]

inv_cov = inv_cov[:,mask_s][mask_s, :]
cov = cov[:,mask_s][mask_s, :]
print(jnp.diag(inv_cov))

k_edges = jnp.arange(1e-2, k_ny, 0.005)
kp, pk, modes = powspec_vec(delta_v, box_size, k_edges) 
dk = kp[1] - kp[0]
pk_var = estimate_pk_variance(kp, pk[:,0], box_size, 0, dk)


k_edges = s_edges_conv(n_bins, box_size, s_edges)
s_centers, xiv, modes = xi_vec(delta_v, box_size, k_edges)
cov = estimate_xi_covariance(s_centers, kp, pk, 3., pk_var, dk)
print(cov.shape)
inv_cov = jnp.array(np.linalg.inv(cov))

print(jnp.diag(inv_cov)); 

window = jnp.ones(4) / 4
nuisance_s = s_centers[:, None]**(-jnp.arange(3)[None, ::-1])
design_matrix = nuisance_s.T.dot(nuisance_s)
print(design_matrix)

def model_curve(params, s):
    alpha, Sigma_nl, B, c, a1, a2, a3 = params
    xi_model = jnp.interp(s * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    return (B**2 * xi_model + a1 / s**2 + a2 / s + a3)
def model_curve_nuisance(params, s, xi_obs, s_obs):
    alpha, Sigma_nl, B, c = params
    xi_model = jnp.interp(s_obs * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    A_obs = xi_obs - xi_model * B**2
    vector = nuisance_s.T.dot(A_obs)
    
    a_params = jax.scipy.linalg.solve(design_matrix, vector, sym_pos=True)

    xi_model = jnp.interp(s * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    return (B**2 * xi_model + a_params.dot((s[:, None]**(-jnp.arange(3)[None,::-1])).T))

def chisq(params, s_obs, xi_obs):
    alpha, Sigma_nl, B, c, a1, a2, a3 = params
    xi_model = jnp.interp(s_obs * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    e =  (B**2 * xi_model + a1 / s_obs**2 + a2 / s_obs + a3) - xi_obs
    return 0.5 * e.T.dot(inv_cov.dot(e)) + alpha_lambda * (alpha - 1.)**2
def chisq_nuisance(params, s_obs, xi_obs):
    # Getting nuisance like this may cause OOM jax errors, set
    # export XLA_PYTHON_CLIENT_PREALLOCATE=false to solve it
    alpha, Sigma_nl, B, c = params
    xi_model = jnp.interp(s_obs * alpha, sarr, template_xi(Sigma_nl, c, sarr, k, plin, plin_nw, smooth_a))
    A_obs = xi_obs - xi_model * B**2
    vector = nuisance_s.T.dot(A_obs)
    a_params = jax.scipy.linalg.solve(design_matrix, vector, sym_pos=True)
    e =  A_obs - a_params.dot(nuisance_s.T)
    return 0.5 * e.T.dot(inv_cov.dot(e))
def neg_loglike(params, s_obs, xi_obs):
    return chisq(params, s_obs, xi_obs)
    #return chisq_nuisance(params, s_obs, xi_obs)

neg_F = jax.hessian(neg_loglike)
score_func = jax.grad(neg_loglike)



params = (1.03, 3., 1., 200., 2.7, -0.2, 0.)
#params = (1.03, 3., 1., 200.)


fisher = jnp.array(neg_F(params, s_centers, xiv[:,0]))
#print("Initial Fisher information: ", jnp.diag(fisher))
#print("Initial scores: ", - jnp.array(score_func(params, s_centers, xiv[:,0])))
fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
std = jnp.sqrt(jnp.diag(cov))
ax[0].errorbar(s_centers, s_centers**2*xiv[:,0], yerr=s_centers**2*std, lw=0, elinewidth=2, marker='o')


ax[1].plot(k, plin_nw)
ax[1].plot(k, plin)
ax[1].format(xscale='log', yscale='log')
fig.savefig("plots/radius_cut.png", dpi=300)



from jax.experimental import optimizers

learning_rate = 0.5e-1
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)
@jax.jit
def step(step, opt_state):
    value, grads = jax.value_and_grad(neg_loglike)(get_params(opt_state), s_centers, xiv[:,0])
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 1000
print("Training...", flush=True)
s = time.time()
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)
print(f"\tDone in {time.time() - s} s", flush=True)
    
params = get_params(opt_state)
print("Final chisq: ", neg_loglike(params, s_centers, xiv[:,0]))
fisher = jnp.array(neg_F(params, s_centers, xiv[:,0]))
print("Final parameters:\n ", jnp.array(params))
print("Final Fisher information:\n ", jnp.diag(fisher))
print("Final scores:\n ", - jnp.array(score_func(params, s_centers, xiv[:,0])))
xi_model = model_curve(params, s_centers)
#xi_model = model_curve_nuisance(params, s_centers, xiv[:,0], s_centers)
ax[0].plot(s_centers, s_centers**2*xi_model)
fig.savefig("plots/radius_cut.png", dpi=300)



param_cov = np.linalg.inv(fisher)
print("Final parameter variances:\n ", jnp.sqrt(jnp.diag(param_cov)))


pk_temp = template_pk(params[1], params[3], k**2, plin, plin_nw)
k_edges = jnp.arange(1e-2, k_ny, 0.001)
kp, pk, modes = powspec_vec(delta_v, box_size, k_edges) 

ax[1].plot(k, pk_temp, label='template')
ax[1].plot(kp, pk[:,0], label='measured')
ax[1].legend(loc='bottom')

s_edges = jnp.linspace(0, 200, 41)
k_edges = s_edges_conv(n_bins, box_size, s_edges)
s_centers, xiv, modes = xi_vec(delta_v, box_size, k_edges)
ax[0].errorbar(s_centers, s_centers**2*xiv[:,0], zorder=0)

#from scipy import signal
#window = signal.windows.gaussian(5, std=0.1)
#ax[0].errorbar(s_centers, s_centers**2*jnp.convolve(xiv[:,0], window, mode='same'))
ax[0].format(ylim=(-20, 30))



fig.savefig("plots/radius_cut.png", dpi=300)