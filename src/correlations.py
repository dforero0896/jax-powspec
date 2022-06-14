from numpy.core.defchararray import _join_dispatcher
import jax
import jax.numpy as jnp
from jax.experimental import loops


@jax.jit
def powspec_vec(delta, box_size, k_edges):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    

    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    
    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    delta_k *= correction_function
    delta2 = (delta_k * delta_k.conj()).real
      
    kedges = k_edges / kF
    
    
    Pk3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten())
    Pk3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D, _ = jnp.histogram(k.flatten(), bins = kedges)
    Pk3D     = jnp.zeros((Pk3D_mono.shape[0], 3), dtype=jnp.float32)
    Pk3D = Pk3D.at[:,0].set(Pk3D_mono/Nmodes3D * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,1].set(Pk3D_quad/Nmodes3D * 5. * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,2].set(Pk3D_hexa/Nmodes3D * 9. * (box_size/dims**2)**3)
    
    k3D = 0.5 * (kedges[1:] + kedges[:-1]) * kF

    return k3D, Pk3D, Nmodes3D



def powspec_vec_fundamental(delta, box_size):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    

    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_index = jnp.int32(k)
    
    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    delta_k *= correction_function
    delta2 = (delta_k * delta_k.conj()).real
      
    k3D = jnp.zeros(kmax+1, dtype=jnp.float32)
    Pk3D     = jnp.zeros((kmax+1,3), dtype=jnp.float32)
    Pkphase  = jnp.zeros(kmax+1, dtype=jnp.float32)
    Nmodes3D = jnp.zeros(kmax+1, dtype=jnp.float32)

    k3D = k3D.at[k_index.flatten()].set(k.flatten())
    Pk3D = Pk3D.at[(k_index.flatten(), 0)].add(delta2.flatten())
    Pk3D = Pk3D.at[(k_index.flatten(), 1)].add(delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D = Pk3D.at[(k_index.flatten(), 2)].add(delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D = Nmodes3D.at[k_index.flatten()].add(1.)
    
    k3D  = k3D[1:]
    Nmodes3D = Nmodes3D[1:]
    Pk3D = Pk3D[1:,:]
    Pkphase = Pkphase[1:]
    k3D = k3D.at[:].set(k3D/Nmodes3D * kF)
    
    Pk3D = Pk3D.at[:,0].set(Pk3D[:,0]/Nmodes3D * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,1].set(Pk3D[:,1]/Nmodes3D * 5. * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,2].set(Pk3D[:,2]/Nmodes3D * 9. * (box_size/dims**2)**3)
    

    return k3D, Pk3D, Nmodes3D
    
    
@jax.jit
def xi_vec(delta, box_size, s_edges):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    k_edges =  kF * s_edges * dims / box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    

    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    
    
    
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    
    delta_k *= correction_function
    delta2 = delta_k * delta_k.conj()
    
    delta_xi = jnp.fft.irfftn(delta2, delta.shape) 
    
    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    #k_index = jnp.int32(k)
        
    k_par = kz
    mu = k_par / k
    mu2 = (mu**2).flatten()
    
        
    kedges = k_edges / kF
    
    
    
    xi3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten())
    xi3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (3.0*mu2-1.0)/2.0)
    xi3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D, _ = jnp.histogram(k.flatten(), bins = kedges)
    Nmodes3D = jnp.where(Nmodes3D == 0., jnp.inf, Nmodes3D)
    xi3D     = jnp.zeros((xi3D_mono.shape[0], 3), dtype=jnp.float32)
    xi3D = xi3D.at[:,0].set(xi3D_mono/Nmodes3D  / dims**3)
    xi3D = xi3D.at[:,1].set(xi3D_quad/Nmodes3D * 5. / dims**3)
    xi3D = xi3D.at[:,2].set(xi3D_hexa/Nmodes3D * 9. / dims**3)
    
    r3D = 0.5 * (kedges[1:] + kedges[:-1]) * (box_size * 1.0 / dims)
  
    
    return r3D, xi3D, Nmodes3D
    
    

def xi_vec_fundamental(delta, box_size):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    

    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    
    
    
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    
    delta_k *= correction_function
    delta2 = delta_k * delta_k.conj()
    
    delta_xi = jnp.fft.irfftn(delta2, delta.shape) 
    
    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_index = jnp.int32(k)
        
    k_par = kz
    mu = k_par / k
    mu2 = (mu**2).flatten()
    
    r3D = jnp.zeros(kmax+1, dtype=jnp.float32)
    xi3D     = jnp.zeros((kmax+1,3), dtype=jnp.float32)
    Nmodes3D = jnp.zeros(kmax+1, dtype=jnp.float32)
    
    r3D = r3D.at[k_index.flatten()].add(k.flatten())
    xi3D = xi3D.at[(k_index.flatten(), 0)].add(delta_xi.flatten())
    xi3D = xi3D.at[(k_index.flatten(), 1)].add(delta_xi.flatten() * (3.0*mu2-1.0)/2.0)
    xi3D = xi3D.at[(k_index.flatten(), 2)].add(delta_xi.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D = Nmodes3D.at[k_index.flatten()].add(1.)
    
    r3D  = r3D[1:]
    Nmodes3D = Nmodes3D[1:]
    xi3D = xi3D[1:,:]
    
    
    r3D = r3D.at[:].set(r3D/Nmodes3D * (box_size * 1.0 / dims))
    xi3D = xi3D.at[:,0].set(xi3D[:,0]/Nmodes3D * (1. / dims**3))
    xi3D = xi3D.at[:,1].set(xi3D[:,1]/Nmodes3D * 5. * (1. / dims**3))
    xi3D = xi3D.at[:,2].set(xi3D[:,2]/Nmodes3D * 9. * (1. / dims**3))
      
    

    
    return r3D, xi3D, Nmodes3D
@jax.jit
def xi_vec_coords(dims, box_size, k_edges):
    kF = 2.0*jnp.pi/box_size
    kedges = k_edges / kF
    r3D = 0.5 * (kedges[1:] + kedges[:-1]) * (box_size * 1.0 / dims)
    return r3D

@jax.jit
def s_edges_conv(dims, box_size, s_edges):
    kF = 2.0*jnp.pi/box_size
    return kF * s_edges * dims / box_size

@jax.jit
def kaiser_power_spectrum_integration(power_spectrum, mu_edges, bias, growth_rate):
    
    
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = mu[1] - mu[0]
    kaiser_factor = (bias + growth_rate * mu**2)**2
    mono = power_spectrum[:,None] * kaiser_factor[None, :]
    print(mono.shape)
    quad = (mono * (2 * 2 + 1 ) * 0.5 * (3. * mu**2 - 1.)[None,:]).sum(axis=1) * dmu
    hexa = (mono * (2 * 4 + 1 ) * 0.125 * (35. * mu**4 - 30. * mu**2 + 3.)[None,:]).sum(axis=1) * dmu
    mono = mono.sum(axis=1) * dmu
    return mono, quad, hexa
@jax.jit
def kaiser_power_spectrum(power_spectrum, bias, growth_rate):

    """Eq. 2.1 in https://arxiv.org/pdf/1610.07785.pdf"""
    
    beta = growth_rate / bias  
    mono = power_spectrum * (bias**2 / 15.) * (15. + 10 * beta + 3 * beta**2)
    quad = power_spectrum * (4. / 21) * growth_rate * (7 * bias + 3 * growth_rate)
    hexa = power_spectrum * (8. / 35) * growth_rate**2
    
    return mono, quad, hexa
@jax.jit
def kaiser_fog_power_spectrum_integrate(power_spectrum, k, mu_edges, bias, growth_rate, sigma):

    """Check https://arxiv.org/pdf/1209.3771.pdf for expansion of the Pk and other FOG kernels"""
    
    
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    kaiser_factor = (bias + growth_rate * mu**2)**2
    mono = power_spectrum[:,None] * kaiser_factor[None, :] * jnp.exp(-k[:,None]**2 * mu[None,:]**2 * sigma**2)
    quad = (mono * (2 * 2 + 1 ) * 0.5 * (3. * mu**2 - 1.)[None,:]).sum(axis=1)
    hexa = (mono * (2 * 4 + 1 ) * 0.125 * (35. * mu**4 - 30. * mu**2 + 2.)[None,:]).sum(axis=1)
    mono = mono.sum(axis=1)
    return mono, quad, hexa

@jax.jit
def estimate_pk_variance(k, pk, box_size, shot_noise, dk):
    """
    Must use linear k bins.
    Eq. 17 in https://arxiv.org/pdf/2109.15236.pdf
    """
    return (2*jnp.pi)**3 / box_size**3 * (2 * (pk + shot_noise)**2 / (4*jnp.pi*k**2 * dk))
@jax.jit
def estimate_xi_covariance(s, k, pk, a_damp, pk_variance, dk):
    """
    Estimation of xi covariance from estimation of Pk variance.
    TODO: Add function to estimate from covariance instead.
    """
    ks = k[:,None] * s[None,:]
    damp = jnp.exp(-k**2*a_damp**2)
    j0 = jnp.sin(ks) / ks
    j0sq = jnp.einsum('ia,ib->iab', j0, j0)
    print(j0sq.shape)
    factor = dk**2 * k**4 * pk_variance * damp**2 / (2*jnp.pi**2)**2
    cov_conf = jnp.nansum(factor[:,None,None] * j0sq, axis=0)
    return cov_conf

@jax.jit
def bispec(delta, box_size, k1, k2, theta):

    dims = delta.shape[0]
    middle = dims // 2
    dims2 = dims**2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))

    bins = theta.shape[0]
    bins      = theta.shape[0]
    k_all     = jnp.zeros(bins+2)
    Pk        = jnp.zeros(bins+2)
    k3        = jnp.sqrt((k2*jnp.sin(theta))**2 + (k2*jnp.cos(theta)+k1)**2)
    k_all = k_all.at[0].set(k1)
    k_all = k_all.at[1].set(k2)
    k_all = k_all.at[2:].set(k3)
    
    k_min = (k_all - kF) / kF
    k_max = (k_all + kF) / kF
    
    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    delta_k *= correction_function
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_index = jnp.int32(k)
    

    #ID = dims2*kxx + dims*kyy + kzz
    id_mask = jnp.logical_and(k[:,:,:,None] >= k_min[None, None, None, :], k[:,:,:,None] < k_max[None, None, None, :])
    
    #delta1_k = jnp.where(id_mask[:,:,:,0], delta_k, 0.)
    #I1_k = jnp.where(id_mask[:,:,:,0], 1., 0.)
    delta1_k = id_mask[:,:,:,0] * delta_k
    I1_k = id_mask[:,:,:,0]

    delta1 = jnp.fft.irfftn(delta1_k, delta.shape)
    I1 = jnp.fft.irfftn(I1_k, delta.shape)

    del delta1_k; del I1_k


    # Compute Pk(k1)
    
    Pk = Pk.at[0].set(0.)
    pairs = 0.
    Pk = Pk.at[0].add(jnp.nansum(delta1**2))
    pairs += (I1**2).sum()
    
    Pk = Pk.at[0].set((Pk[0] / pairs) * (box_size / dims**2)**3)
    
    
    # Fill delta2_k array and compute delta2

    #delta2_k = jnp.where(id_mask[:,:,:,1], delta_k, 0.)
    #I2_k = jnp.where(id_mask[:,:,:,1], 1., 0.)
    delta2_k = id_mask[:,:,:,1] * delta_k
    I2_k = id_mask[:,:,:,1]

    delta2 = jnp.fft.irfftn(delta2_k, delta.shape)
    I2 = jnp.fft.irfftn(I2_k, delta.shape)

    del delta2_k; del I2_k
    
    
    # Compute Pk(k2)
    
    Pk = Pk.at[1].set(0.)
    pairs = 0.
    Pk = Pk.at[1].add(jnp.nansum(delta2**2))
    pairs += (I2**2).sum()
    
    Pk = Pk.at[1].set((Pk[1] / pairs) * (box_size / dims**2)**3)
        

    def scan_func(carry, x):
        # x is the iteration index
        id_mask, delta_k, box_size, Pk, delta1, delta2, I1, I2 = carry
        dims = delta_k.shape[0]
        #delta3_k = jnp.where(id_mask[:,:,:,x+2], delta_k, 0.)
        #I3_k = jnp.where(id_mask[:,:,:,x+2], 1., 0.)
        delta3_k = id_mask[:,:,:,x+2] * delta_k
        I3_k = id_mask[:,:,:,x+2]

        delta3 = jnp.fft.irfftn(delta3_k, (dims, dims, dims))
        I3 = jnp.fft.irfftn(I3_k, (dims, dims, dims))

        del delta3_k; del I3_k

        Pk_val = 0.
        pairs = 0.
        Pk_val += jnp.nansum(delta3**2)
        pairs += jnp.nansum(I3**2)

        Pk_val = (Pk_val / pairs) * (box_size / dims**2)**3
        Pk = Pk.at[x+2].set(Pk_val)
        
        B_val = 0.
        triangles = 0.
        B_val += jnp.nansum(delta1*delta2*delta3)
        triangles += (I1*I2*I3).sum()

        B_val = (B_val / triangles) * (box_size**2 / dims**3)**3
        Q_val = B_val / (Pk[0] * Pk[1] + Pk[0] * Pk_val + Pk[1] * Pk_val)


        return carry, jnp.array((Pk_val, B_val, Q_val))
    xs = jnp.arange(0, bins, 1)
    _, PBQ = jax.lax.scan(scan_func, (id_mask, delta_k, box_size, Pk, delta1, delta2, I1, I2), xs)

    Pk = Pk.at[2:].set(PBQ[:,0])
    
    
    return k_all, Pk, theta, PBQ[:,1], PBQ[:,2]

@jax.jit
def compute_all_correlations(delta, box_size, s_edges, k_edges, k1, k2, theta):

    dims = delta.shape[0]
    middle = dims // 2
    dims2 = dims**2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))

    bins = theta.shape[0]
    k_all = jnp.zeros(bins+2)
    Pk = jnp.zeros(bins+2)
    k3 = jnp.sqrt((k2*jnp.sin(theta))**2 + (k2*jnp.cos(theta)+k1)**2)
    k_all = k_all.at[0].set(k1)
    k_all = k_all.at[1].set(k2)
    k_all = k_all.at[2:].set(k3)
    k_min = (k_all - kF) / kF
    k_max = (k_all + kF) / kF
    
    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    delta_k *= cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_index = jnp.int32(k)


    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    delta2 = (delta_k * delta_k.conj()).real
      
    kedges = k_edges / kF
    
    
    Pk3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten())
    Pk3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D_pk, _ = jnp.histogram(k.flatten(), bins = kedges)
    Pk3D     = jnp.zeros((Pk3D_mono.shape[0], 3), dtype=jnp.float32)
    Pk3D = Pk3D.at[:,0].set(Pk3D_mono/Nmodes3D_pk * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,1].set(Pk3D_quad/Nmodes3D_pk * 5. * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,2].set(Pk3D_hexa/Nmodes3D_pk * 9. * (box_size/dims**2)**3)
    k3D = 0.5 * (kedges[1:] + kedges[:-1]) * kF

    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)        
    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    
    delta_xi = jnp.fft.irfftn(delta2, delta.shape) 
    k_edges =  kF * s_edges * dims / box_size
    kedges = k_edges / kF

    xi3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten())
    xi3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (3.0*mu2-1.0)/2.0)
    xi3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D_xi, _ = jnp.histogram(k.flatten(), bins = kedges)
    Nmodes3D_xi = jnp.where(Nmodes3D_xi == 0., jnp.inf, Nmodes3D_xi)
    xi3D     = jnp.zeros((xi3D_mono.shape[0], 3), dtype=jnp.float32)
    xi3D = xi3D.at[:,0].set(xi3D_mono/Nmodes3D_xi  / dims**3)
    xi3D = xi3D.at[:,1].set(xi3D_quad/Nmodes3D_xi * 5. / dims**3)
    xi3D = xi3D.at[:,2].set(xi3D_hexa/Nmodes3D_xi * 9. / dims**3)
    r3D = 0.5 * (kedges[1:] + kedges[:-1]) * (box_size * 1.0 / dims)

    del delta_xi
    del delta2



    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))

    k = jnp.sqrt(kx**2 + ky**2 + kz**2)


    id_mask = jnp.logical_and(k[:,:,:,None] >= k_min[None, None, None, :], k[:,:,:,None] < k_max[None, None, None, :])
    
    delta1_k = jnp.where(id_mask[:,:,:,0], delta_k, 0.)
    #I1_k = jnp.where(id_mask[:,:,:,0], 1., 0.)
    I1_k = id_mask[:,:,:,0]

    delta1 = jnp.fft.irfftn(delta1_k, delta.shape)
    I1 = jnp.fft.irfftn(I1_k, delta.shape)

    del delta1_k; del I1_k


    # Compute Pk(k1)
    
    Pk = Pk.at[0].set(0.)
    pairs = 0.
    Pk = Pk.at[0].add(jnp.nansum(delta1**2))
    pairs += (I1**2).sum()
    
    Pk = Pk.at[0].set((Pk[0] / pairs) * (box_size / dims**2)**3)
    

    # Fill delta2_k array and compute delta2

    delta2_k = jnp.where(id_mask[:,:,:,1], delta_k, 0.)
    #I2_k = jnp.where(id_mask[:,:,:,1], 1., 0.)
    I2_k = id_mask[:,:,:,1]

    delta2 = jnp.fft.irfftn(delta2_k, delta.shape)
    I2 = jnp.fft.irfftn(I2_k, delta.shape)

    del delta2_k; del I2_k
    
    
    # Compute Pk(k2)
    Pk = Pk.at[1].set(0.)
    pairs = 0.
    Pk = Pk.at[1].add(jnp.nansum(delta2**2))
    pairs += (I2**2).sum()
    
    Pk = Pk.at[1].set((Pk[1] / pairs) * (box_size / dims**2)**3)
    

    def scan_func(carry, x):
        # x is the iteration index
        id_mask, delta_k, box_size, Pk, delta1, delta2, I1, I2 = carry
        dims = delta_k.shape[0]
        delta3_k = jnp.where(id_mask[:,:,:,x+2], delta_k, 0.)
        #I3_k = jnp.where(id_mask[:,:,:,x+2], 1., 0.)
        I3_k = id_mask[:,:,:,x+2]

        delta3 = jnp.fft.irfftn(delta3_k, (dims, dims, dims))
        I3 = jnp.fft.irfftn(I3_k, (dims, dims, dims))

        del delta3_k; del I3_k

        Pk_val = 0.
        pairs = 0.
        Pk_val += jnp.nansum(delta3**2)
        pairs += jnp.nansum(I3**2)

        Pk_val = (Pk_val / pairs) * (box_size / dims**2)**3
        Pk = Pk.at[x+2].set(Pk_val)

        B_val = 0.
        triangles = 0.
        B_val += jnp.nansum(delta1*delta2*delta3)
        triangles += (I1*I2*I3).sum()

        B_val = (B_val / triangles) * (box_size**2 / dims**3)**3
        Q_val = B_val / (Pk[0] * Pk[1] + Pk[0] * Pk_val + Pk[1] * Pk_val)


        return carry, jnp.array((Pk_val, B_val, Q_val))
    xs = jnp.arange(0, bins, 1)
    _, PBQ = jax.lax.scan(scan_func, (id_mask, delta_k, box_size, Pk, delta1, delta2, I1, I2), xs)

    
    Pk = Pk.at[2:].set(PBQ[:,0])
    
    return k3D, Pk3D, Nmodes3D_pk, r3D, xi3D, Nmodes3D_xi, k_all, Pk, theta, PBQ[:,1], PBQ[:,2]


@jax.jit
def compute_2pt_correlations(delta, box_size, s_edges, k_edges):

    dims = delta.shape[0]
    middle = dims // 2
    dims2 = dims**2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))

    
    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    

    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, middle+1))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, middle+1))
    kz = jnp.broadcast_to(ki[None, None, :middle+1], (dims, dims, middle+1))
    delta_k *= cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_index = jnp.int32(k)


    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    delta2 = (delta_k * delta_k.conj()).real
      
    kedges = k_edges / kF
    
    
    Pk3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten())
    Pk3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D_pk, _ = jnp.histogram(k.flatten(), bins = kedges)
    Pk3D     = jnp.zeros((Pk3D_mono.shape[0], 3), dtype=jnp.float32)
    Pk3D = Pk3D.at[:,0].set(Pk3D_mono/Nmodes3D_pk * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,1].set(Pk3D_quad/Nmodes3D_pk * 5. * (box_size/dims**2)**3)
    Pk3D = Pk3D.at[:,2].set(Pk3D_hexa/Nmodes3D_pk * 9. * (box_size/dims**2)**3)
    k3D = 0.5 * (kedges[1:] + kedges[:-1]) * kF

    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)        
    k_par = kz
    mu = jnp.where(k == 0., 0., k_par / k)
    mu2 = (mu**2).flatten()
    
    delta_xi = jnp.fft.irfftn(delta2, delta.shape) 
    k_edges =  kF * s_edges * dims / box_size
    kedges = k_edges / kF

    xi3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten())
    xi3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (3.0*mu2-1.0)/2.0)
    xi3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta_xi.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D_xi, _ = jnp.histogram(k.flatten(), bins = kedges)
    Nmodes3D_xi = jnp.where(Nmodes3D_xi == 0., jnp.inf, Nmodes3D_xi)
    xi3D     = jnp.zeros((xi3D_mono.shape[0], 3), dtype=jnp.float32)
    xi3D = xi3D.at[:,0].set(xi3D_mono/Nmodes3D_xi  / dims**3)
    xi3D = xi3D.at[:,1].set(xi3D_quad/Nmodes3D_xi * 5. / dims**3)
    xi3D = xi3D.at[:,2].set(xi3D_hexa/Nmodes3D_xi * 9. / dims**3)
    r3D = 0.5 * (kedges[1:] + kedges[:-1]) * (box_size * 1.0 / dims)
    
    return k3D, Pk3D, Nmodes3D_pk, r3D, xi3D


