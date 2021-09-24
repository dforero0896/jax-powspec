import numpy as np
from Corrfunc.theory.xi import xi as xi_corrfunc
import matplotlib as mpl
mpl.use("Agg")
import proplot as pplt
import os, sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
sys.path.insert(0, "/home/astro/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz

bins = np.linspace(1e-5, 200, 41)
if not os.path.isfile("data/corrfunc_xi.npy"):
    data = np.load("data/lognormal_nobao_corrected.npy")
    box_size = 1000.
    results = xi_corrfunc(box_size, 32, bins, data[:,0], data[:,1], data[:,2], output_ravg=True, verbose=True)
    np.save("data/corrfunc_xi.npy", results)
else:
    results = np.load("data/corrfunc_xi.npy")

fig, ax = pplt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)

ax[0].plot(results['ravg'], results['ravg']**2 * results['xi'], label="Corrfunc")

fig.savefig("plots/corrfunc_xi.png", dpi=300)

bias = 2.
redshift = 0.55
z = redshift
klin = np.logspace(-3, 0, 2048)
plin = bias**2 * jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu)
s, xi = xicalc_trapz(klin, plin, 2., results['ravg'])
ax[0].plot(s, s**2*xi, label='Linear (targ.)', ls='--')
ax[0].legend(loc='top')
ax[0].format(xlabel='$s$', ylabel = r'$s^2\xi$')



fig.savefig("plots/corrfunc_xi.png", dpi=300)


