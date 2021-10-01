import numpy as np
from Corrfunc.theory.xi import xi as xi_corrfunc
from Corrfunc.theory.DDsmu import DDsmu
import matplotlib as mpl
mpl.use("Agg")
import proplot as pplt
import os, sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
sys.path.insert(0, "/home/astro/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import jax.numpy as jnp
from scipy.special import legendre
from src.correlations import kaiser_power_spectrum


def pair_counts_to_mp(pair_counts, sbin_arr, n_mu_bin, poles):
    mu_edges = np.linspace(0,1,n_mu_bin+1)
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    multipoles = np.zeros((sbin_arr.shape[0] - 1, len(poles)))
    
    for i, l in enumerate(poles):
        factor = (2 * l + 1) / n_mu_bin
        #factor = (2 * l + 1) * (mu[1] - mu[0])
        multipoles[:, i] = (pair_counts.reshape(sbin_arr.shape[0] - 1, n_mu_bin) * factor * legendre(l)(mu)[None, :]).sum(axis=1)
    return multipoles

def rr_analytic2d(bin_low_bound, bin_high_bound, box_size, nmu_bins=40):
    volume = 4 * np.pi * (bin_high_bound**3 - bin_low_bound**3) / 3
    normed_volume = volume / box_size **3
    mu_edges = np.linspace(0,1,nmu_bins+1)
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mono = normed_volume[:,None] / (nmu_bins*np.ones(nmu_bins)[None,:])
    
    return mono


bins = np.linspace(1e-5, 200, 41)
box_size = 1000.
nmu_bins = 40
data = np.load("data/lognormal_nobao_multi_corrected.npy")
if not os.path.isfile("data/corrfunc_xi.npz"):
    
    print("Computing pair counts ...", flush=True)
    results = DDsmu(1, 32, bins, 1., 40,
                  data[:,0], data[:,1], data[:,2],
                  output_savg=True, boxsize=box_size, periodic=True)
    norm = data.shape[0] * (data.shape[0] - 1.)
    np.savez("data/corrfunc_xi.npz", results=results, norm=norm)
else:
    print("Loading pair counts ...", flush=True)
    _results = np.load("data/corrfunc_xi.npz")
    results = _results['results']
    norm = _results['norm']

rr = rr_analytic2d(bins[:-1], bins[1:], box_size, nmu_bins=nmu_bins)
xi = ((results['npairs'].reshape(bins.shape[0] - 1, nmu_bins) / norm) - rr) / rr
multipoles = pair_counts_to_mp(xi, bins, nmu_bins, [0,2,4])


fig, ax = pplt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
s_center = 0.5 * (bins[1:] + bins[:-1])
ax[0].plot(s_center, s_center**2 * multipoles[:,0], label="Corrfunc")
ax[1].plot(s_center, s_center**2 * multipoles[:,1], label="Corrfunc")
ax[2].plot(s_center, s_center**2 * multipoles[:,2], label="Corrfunc")

fig.savefig("plots/corrfunc_xi.png", dpi=300)

bias = 2.
redshift = 0.55
z = redshift
cosmo = jc.Planck15()
growth_rate = jc.background.growth_rate(cosmo, jnp.array([1. / (1. + z)])).squeeze()
shot_noise = box_size**3 / data.shape[0]
klin = np.logspace(-3, 0, 2048)
plin = jc.power.linear_matter_power(cosmo, klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) 
pk0, pk2, pk4 = kaiser_power_spectrum(plin, bias, growth_rate)
s, xi0, _, _ = xicalc_trapz(klin, pk0 + shot_noise, 2., s_center)
s, _, xi2, _ = xicalc_trapz(klin, pk2, 2., s_center)
s, _, _, xi4 = xicalc_trapz(klin, pk4, 2., s_center)
ax[0].plot(s, s**2*xi0, label='Linear (targ.)', ls='--')
ax[1].plot(s, s**2*xi2, label='Linear (targ.)', ls='--')
ax[2].plot(s, s**2*xi4, label='Linear (targ.)', ls='--')
ax[0].legend(loc='top')
ax.format(xlabel='$s$', ylabel = r'$s^2\xi$')



fig.savefig("plots/corrfunc_xi.png", dpi=300)


