import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/daniel/OneDrive/Research/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import numpy as np
import pandas as pd
import proplot as pplt

from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, kaiser_power_spectrum, kaiser_power_spectrum_integration

z = 0.
bias = 1
cosmo = jc.Planck15()
growth_rate = jc.background.growth_rate(cosmo, jnp.array([1. / (1. + z)])).squeeze()

fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
klin = jnp.logspace(-2, 0, 2048)
s = jnp.linspace(0, 200, 1000)
plin = jc.power.linear_matter_power(cosmo, klin, a=1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu) 
s, xi0, xi2, xi4 = xicalc_trapz(klin, plin, 2., s)
ax[0].plot(s, s**2*xi0, label='linear')
#ax[1].plot(s, s**2*xi2, label='linear')
#ax[2].plot(s, s**2*xi4, label='linear')



pk0, pk2, pk4 = kaiser_power_spectrum(plin, bias, growth_rate)
ax[3].plot(klin, klin*pk0, label='kaiser mult', ls='-')
ax[4].plot(klin, klin*pk2, label='kaiser mult', ls='-')
ax[5].plot(klin, klin*pk4, label='kaiser mult', ls='-')

mu_edges = jnp.linspace(0,1,2048+1)
pk0, pk2, pk4 = kaiser_power_spectrum_integration(plin, mu_edges, bias, growth_rate)
ax[3].plot(klin, klin*pk0, label='1d to mult 2048', ls='--')
ax[4].plot(klin, klin*pk2, label='1d to mult 2048', ls='--')
ax[5].plot(klin, klin*pk4, label='1d to mult 2048', ls='--')

ax[4].legend(loc='bottom')


s, _, xi2, _ = xicalc_trapz(klin, pk2, 2., s)
s, _, _, xi4 = xicalc_trapz(klin, pk4, 2., s)
ax[1].plot(s, s**2*xi2, label='transf', ls='--')
ax[2].plot(s, s**2*xi4, label='transf', ls='--')




ax[1,1:].format(yscale='linear', xscale='log')
ax[1,0].format(yscale='linear', xscale='log')
ax[1].legend(loc='bottom')
ax[2].legend(loc='bottom')

fig.savefig("plots/plin_multipoles.png", dpi=300)

