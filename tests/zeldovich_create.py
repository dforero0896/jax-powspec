import numpy as np
import density_field_library as DFL
import Pk_library as PKL
import jax_cosmo as jc
import jax.numpy as jnp
import proplot as pplt

grid              = 450    #grid size
BoxSize           = 1000.0 #Mpc/h
seed              = 42      #value of the initial random seed
Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
threads           = 8      #number of openmp threads
verbose           = True   #whether to print some information
cosmo = jc.Planck15()
z = 0.55
b1 = 2.
k = np.array(jnp.logspace(-3., 1., 1000))
Pk = np.array(b1**2 * jc.power.linear_matter_power(cosmo, k, a = 1. / (1 + z), transfer_fn=jc.transfer.Eisenstein_Hu))

# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
                              BoxSize, threads, verbose)

delta = df_3D / df_3D.mean()
delta -= 1.
axis=0
MAS=None
Pk_m = PKL.Pk(delta, BoxSize, axis, MAS, threads)

fig, ax = pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
ax[0].plot(k, k*Pk)
ax[0].plot(Pk_m.k3D, Pk_m.k3D*Pk_m.Pk[:,0])
ax[1].plot(Pk_m.k3D, Pk_m.k3D*Pk_m.Pk[:,1])
ax[2].plot(Pk_m.k3D, Pk_m.k3D*Pk_m.Pk[:,2])
ax[0,:].format(xscale='log')

fig.savefig("plots/zeldovich_create.png", dpi=300)

CF_m     = PKL.Xi(delta, BoxSize, MAS, axis, threads)
mask = CF_m.r3D < 200

ax[3].plot(CF_m.r3D[mask], CF_m.r3D[mask]*CF_m.xi[mask,0])
ax[4].plot(CF_m.r3D[mask], CF_m.r3D[mask]*CF_m.xi[mask,1])
ax[5].plot(CF_m.r3D[mask], CF_m.r3D[mask]*CF_m.xi[mask,2])

fig.savefig("plots/zeldovich_create.png", dpi=300)

delta_k = np.fft.fftn(delta)
k = np.fft.fftfreq(grid, d=BoxSize / grid) * 2 * np.pi
ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None, None,:]**2
delta_k/=ksq
del ksq
psi_x = 1j * k[:,None,None] * delta_k