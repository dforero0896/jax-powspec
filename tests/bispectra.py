import numpy as np
import proplot as pplt
import MAS_library as MASL
import Pk_library as PKL
import jax_cosmo as jc

grid    = 256    #the 3D field will have grid x grid x grid voxels
BoxSize = 1000.0 #Mpc/h ; size of box
MAS     = 'CIC'  #mass-assigment scheme
verbose = True   #print information on progress

k1      = 0.2    #h/Mpc
k2      = 0.4    #h/Mpc
theta   = np.linspace(0, np.pi, 50) #array with the angles between k1 and k2
threads = 8

data = np.load("data/lognormal_nobao_corrected.npy")
delta = np.zeros((grid,grid,grid), dtype=np.float32)


MASL.MA(data.astype(np.float32), delta, BoxSize, MAS, verbose=verbose)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
BBk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)

fig, ax = pplt.subplots(nrows = 1, ncols=2, sharex=False, sharey=False)

ax[0].plot(BBk.k, BBk.Pk)
ax[0].format(xlabel=r'$k$ $h$/Mpc', ylabel=r'$P(k)$', yscale='log', xscale='log')

ax[1].plot(theta, BBk.B, label=f'$k_1 = {k1},\ k_2={k2}$ $h$/Mpc')
ax[1].format(xlabel=r'$\theta$', ylabel=r'$Q(\theta)$', yscale='log')

fig.savefig("plots/bispectra.png", dpi=300)

redshift = 0.55
b1 = 2.0
cosmo = jc.Planck15()
klin = np.logspace(-3, 0, 2048)
plin = np.array(b1**2 * jc.power.linear_matter_power(jc.Planck15(), klin, a=1. / (1 + redshift), transfer_fn=jc.transfer.Eisenstein_Hu))


theta_theory, Bk_theory = PKL.Bispectrum_theory(klin, plin, k1, k2)

ax[1].plot(theta_theory, Bk_theory)
fig.savefig("plots/bispectra.png", dpi=300)
