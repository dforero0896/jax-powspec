import jax
import jax.numpy as jnp

def get_positions(key, number_particles, bin_size):
    key, subkey = jax.random.split(key)
    Rs = 2 * jax.random.uniform(subkey, shape = (number_particles,3)) - 1
    Rs = jnp.sign(Rs) * (1 - jnp.sqrt(jnp.abs(Rs)))
    
    return Rs * bin_size

def populate_field(rho, n_bins, box_size, key):
    bin_size = box_size / n_bins
    nonzero = (rho != 0).sum()
    sorted_rho = jnp.argsort(rho.ravel())[::-1]#[:nonzero+1]
    
    flat_rho = rho.flatten()[sorted_rho]
    grid_centers = jnp.array(jnp.unravel_index(sorted_rho, (n_bins, n_bins, n_bins))).T.astype(jnp.float32) * bin_size + 0.5 * bin_size
    
    key, subkey = jax.random.split(key)
    number_objects = jax.random.poisson(subkey, flat_rho, shape=flat_rho.shape)
    
    displacements = get_positions(key, number_objects.sum(), bin_size)

    coords = jnp.repeat(grid_centers, number_objects, axis=0) + displacements
    return (coords + box_size) % box_size



def ezmock(params, rho, n_bins, box_size, hubble, growth_rate, displacement_potential, key):

    rho_low, rho_sat = params
    lamda = 10.
     
    
    rho = jnp.clip(rho, 0, rho_sat)
    rho - jnp.where(rho < rho_low, 0., rho)
    key, subkey = jax.random.split(key)
    G = jax.random.normal(subkey, rho.shape) * lamda
    G = jnp.where(G >= 0, 1 + G, jnp.exp(G))
    rho *= G

    # Apply RSD

    
    


    positions = populate_field(rho, n_bins, box_size, key)
    return positions



def zeldovich_rsd_parallel(delta, bias, f, smooth=10.):
    beta = f / bias
    k = jnp.fft.fftfreq(grid, d=BoxSize/grid) * 2 * np.pi
    kr = k * smooth
    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :grid//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:grid//2+1]**2
    delta_displaced = delta.copy() 
    phi_field = jnp.fft.rfftn(delta_displaced) * norm / ksq # Eq. 24 https://arxiv.org/pdf/1504.02591.pdf
    phi_field[0,0,0] = 0.
    los = jnp.array([0,0,1])
    displacement = k[None,None,:grid//2+1]**2 * phi_field #Eq. 25 plane-parallel approx
    delta_displaced -= beta * jnp.fft.irfftn(displacement, delta.shape)
    return delta_displaced
    
    







    


