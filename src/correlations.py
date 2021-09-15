from numpy.core.defchararray import _join_dispatcher
import jax
import jax.numpy as jnp
from jax.experimental import loops

def powspec(delta, box_size, axis):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    
    subs_dims = lambda x: x - dims
    nothing = lambda x : x

    def cic_correction(x, index):
        return jax.lax.cond(x==0, lambda y: 1., lambda y: (y / jnp.sin(y))**index, x)

    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))

    with loops.Scope() as s:
        s.k3D = jnp.zeros(kmax+1, dtype=jnp.float32)
        s.Pk3D     = jnp.zeros((kmax+1,3), dtype=jnp.float32)
        s.Pkphase  = jnp.zeros(kmax+1, dtype=jnp.float32)
        s.Nmodes3D = jnp.zeros(kmax+1, dtype=jnp.float32)

        for kxx in s.range(dims):
            kx = jax.lax.cond(kxx > middle, subs_dims, nothing, kxx)
            kx_correction = cic_correction(prefact * kx, 2)

            for kyy in s.range(dims):
                ky = jax.lax.cond(kyy > middle, subs_dims, nothing, kyy)
                ky_correction = cic_correction(prefact * ky, 2)

                for kzz in s.range(middle + 1):
                    kz = jax.lax.cond(kzz > middle, subs_dims, nothing, kzz)
                    kz_correction = cic_correction(prefact * kz, 2)

                    for _ in s.cond_range(jnp.logical_or(kz == 0, jnp.logical_and(kz==middle, dims%2==0))):
                        for _ in s.cond_range(kx < 0):
                            continue
                        for _ in s.cond_range(jnp.logical_or(kx == 0, jnp.logical_and(kx==middle, dims%2==0))):
                            for _ in s.cond_range(ky < 0):
                                continue
                    k = jnp.sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = jnp.int32(k)
                    k_par = 0
                    k_per = 0
                    """
                    k_par, k_per = jax.lax.cond(axis == 0, lambda tup: (tup[0], jnp.int32(jnp.sqrt(tup[1]**2 + tup[2]**2))),
                                    jax.lax.cond(axis == 1, lambda tup: (tup[1], jnp.int32(jnp.sqrt(tup[0]**2 + tup[2]**2))),
                                    lambda tup: (tup[2], jnp.int32(jnp.sqrt(tup[0]**2 + tup[1]**2))),
                                    (kx, ky, kz)), (kx, ky, kz)
                    )
                    """
                    k_par = kz
                    k_per = jnp.int32(jnp.sqrt(kx**2 + ky**2))
                    mu = 0.
                    mu = jax.lax.cond(k == 0., lambda x: 0., lambda x: k_par / k, mu)
                    mu2 = mu**2

                    k_par = jax.lax.cond(k_par < 0, lambda x: -x, lambda x: x, k_par)

                    correction_factor = kx_correction * ky_correction * kz_correction

                    delta_k = jax.ops.index_update(delta_k, (kxx, kyy, kzz), delta_k[kxx, kyy, kzz] * correction_factor)

                    real = delta_k[kxx, kyy, kzz].real
                    imag = delta_k[kxx, kyy, kzz].imag
                    delta2 = real**2 + imag**2
                    phase = jnp.angle(delta_k[kxx, kyy, kzz])

                    # Pk3D.
                    s.k3D = jax.ops.index_add(s.k3D, k_index, k)
                    s.Pk3D = jax.ops.index_add(s.Pk3D, (k_index, 0), delta2)
                    s.Pk3D = jax.ops.index_add(s.Pk3D, (k_index, 1), delta2*(3.0*mu2-1.0)/2.0)
                    s.Pk3D = jax.ops.index_add(s.Pk3D, (k_index, 2), delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                    s.Pkphase = jax.ops.index_add(s.Pkphase, k_index, phase*phase)
                    s.Nmodes3D = jax.ops.index_add(s.Nmodes3D, k_index, 1.)
                    
        s.k3D  = s.k3D[1:]
        s.Nmodes3D = s.Nmodes3D[1:]
        s.Pk3D = s.Pk3D[1:,:]
        s.Pkphase = s.Pkphase[1:]

        for i in s.range(s.k3D.shape[0]):
            s.k3D = jax.ops.index_update(s.k3D, i, (s.k3D[i]/s.Nmodes3D[i])*kF)
            s.Pk3D = jax.ops.index_update(s.Pk3D, (i,0), (s.Pk3D[i,0]/s.Nmodes3D[i])*(box_size/dims**2)**3)
            s.Pk3D = jax.ops.index_update(s.Pk3D, (i,1), (s.Pk3D[i,1]*5.0/s.Nmodes3D[i])*(box_size/dims**2)**3)
            s.Pk3D = jax.ops.index_update(s.Pk3D, (i,2), (s.Pk3D[i,2]*9.0/s.Nmodes3D[i])*(box_size/dims**2)**3)
            s.Pkphase = jax.ops.index_update(s.Pkphase, i, (s.Pkphase[i]/s.Nmodes3D[i])*(box_size/dims**2)**3)
        mask = s.k3D < kN
        return s.k3D[mask], s.Pk3D[mask], s.Pkphase[mask], s.Nmodes3D[mask]



def powspec_vec(delta, box_size, axis):

    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0*jnp.pi/box_size
    kN = middle*kF
    kmax_par = middle
    prefact = jnp.pi / dims
    kmax_per = jnp.int32(jnp.sqrt(middle**2 + middle**2))
    kmax     = jnp.int32(jnp.sqrt(middle**2 + middle**2 + middle**2))
    print(kmax)

    def cic_correction(x, index):
        return (1. / jnp.sinc(x/jnp.pi))**index
    subs_dims = lambda x: x - dims
    nothing = lambda x : x

    delta_k = jnp.fft.rfftn(delta, axes = (0,1,2))
    print(delta_k.shape)
    
    ki = jnp.fft.fftfreq(dims, d=1./dims).astype(jnp.int32)
    
    
    
    kxx, kyy, kzz = jnp.meshgrid(jnp.arange(dims), jnp.arange(dims), jnp.arange(middle + 1), indexing='ij')
    kx = jax.ops.index_add(kxx, (kxx > middle).nonzero(), - dims)
    ky = jax.ops.index_add(kyy, (kyy > middle).nonzero(), - dims)
    kz = jax.ops.index_add(kzz, (kzz > middle).nonzero(), - dims)
    correction_function = cic_correction(prefact * kx, 2.) * cic_correction(prefact * ky, 2.) * cic_correction(prefact * kz, 2.)
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    #k = jnp.sqrt(ki[:,None,None]**2 + ki[None,:,None]**2 + ki[None,None,:middle+1]**2)
    k_index = jnp.int32(k)
    
    k_par = kz
    mu = k_par / k
    mu2 = (mu**2).flatten()
    delta_k *= correction_function
    delta2 = delta_k * delta_k.conj()
    
   
    k3D = jnp.zeros(kmax+1, dtype=jnp.float32)
    Pk3D     = jnp.zeros((kmax+1,3), dtype=jnp.float32)
    Pkphase  = jnp.zeros(kmax+1, dtype=jnp.float32)
    Nmodes3D = jnp.zeros(kmax+1, dtype=jnp.float32)
    k3D = jax.ops.index_add(k3D, k_index.flatten(), k.flatten())
    Pk3D = jax.ops.index_add(Pk3D, (k_index.flatten(), 0), delta2.flatten())
    Pk3D = jax.ops.index_add(Pk3D, (k_index.flatten(), 1), delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D = jax.ops.index_add(Pk3D, (k_index.flatten(), 2), delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D = jax.ops.index_add(Nmodes3D, k_index.flatten(), 1.)
    k3D  = k3D[1:]
    Nmodes3D = Nmodes3D[1:]
    Pk3D = Pk3D[1:,:]
    Pkphase = Pkphase[1:]
    k3D = jax.ops.index_update(k3D, jax.ops.index[:], k3D/Nmodes3D * kF)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,0], Pk3D[:,0]/Nmodes3D * (box_size/dims**2)**3)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,1], Pk3D[:,1]/Nmodes3D * 5. * (box_size/dims**2)**3)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,2], Pk3D[:,2]/Nmodes3D * 9. * (box_size/dims**2)**3)
    
    kedges = jnp.arange(1e-8, 1, 4e-3) / kF
    
    
    Pk3D_mono, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten())
    Pk3D_quad, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (3.0*mu2-1.0)/2.0)
    Pk3D_hexa, _ = jnp.histogram(k.flatten(), bins = kedges, weights = delta2.flatten() * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
    Nmodes3D, _ = jnp.histogram(k.flatten(), bins = kedges)
    Pk3D     = jnp.zeros((Pk3D_mono.shape[0], 3), dtype=jnp.float32)
    Pkphase     = jnp.zeros((Pk3D_mono.shape[0]), dtype=jnp.float32)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,0], Pk3D_mono/Nmodes3D * (box_size/dims**2)**3)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,1], Pk3D_quad/Nmodes3D * 5. * (box_size/dims**2)**3)
    Pk3D = jax.ops.index_update(Pk3D, jax.ops.index[:,2], Pk3D_hexa/Nmodes3D * 9. * (box_size/dims**2)**3)
    k3D = 0.5 * (kedges[1:] + kedges[:-1]) * kF
    

    mask = k3D < kN
    return k3D[mask], Pk3D[mask], Pkphase[mask], Nmodes3D[mask]
    
    

