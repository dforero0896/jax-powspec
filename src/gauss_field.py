
import numpy as np
import math

def gaussian_field(grid, kf, Pkf, Rayleigh_sampling, 
                      seed, BoxSize):
    """ Implementation taken from Pylians3 """
        

    phase_prefac = 2.0*np.pi
    k_prefac     = 2.0*np.pi/BoxSize
    inv_max_rand = 1.0
    zero = 0. + 1j * 0.
    
    k_bins, middle = len(kf), grid//2

    # define the density field in Fourier space
    delta_k = np.zeros((grid, grid, middle+1), dtype=np.complex64)

    # initialize the random generator
    np.random.seed(seed)
    
    # we make a loop over the indexes of the matrix delta_k(kxx,kyy,kzz)
    # but the vector k is given by \vec{k}=(kx,ky,kz)
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        kxx_m = (grid-kx if (kx>0) else -kx) #index corresponding to -kx

        for kyy in range(grid):
            ky = (kyy-grid if (kyy>middle) else kyy)
            kyy_m = (grid-ky if (ky>0) else -ky) #index corresponding to -ky

            for kzz in range(middle+1):
                kz = (kzz-grid if (kzz>middle) else kzz)

                # find the value of |k| of the mode
                kmod = math.sqrt(kx*kx + ky*ky + kz*kz)*k_prefac
                
                # interpolate to compute P(|k|)
                lmin = 0;  lmax = k_bins-1
                while (lmax-lmin>1):
                    l = (lmin+lmax)//2
                    if kf[l]<kmod:  lmin = l
                    else:           lmax = l
                Pk = ((Pkf[lmax]-Pkf[lmin])/(kf[lmax]-kf[lmin])*\
                      (kmod-kf[lmin]))+Pkf[lmin]           
                Pk = Pk*(grid**2/BoxSize)**3 #remove units

                #generate the mode random phase and amplitude
                phase     = phase_prefac*np.random.random()
                amplitude = inv_max_rand*np.random.random()
                while (amplitude==0.0):   amplitude = inv_max_rand*np.random.random()
                if Rayleigh_sampling==1:  amplitude = math.sqrt(-math.log(amplitude))
                else:                     amplitude = 1.0
                amplitude *= math.sqrt(Pk)
                
                # get real and imaginary parts
                real_part = amplitude*math.cos(phase)
                imag_part = amplitude*math.sin(phase)

                # fill the upper plane of the delta_k array
                if delta_k[kxx,kyy,kzz]==zero:
                    delta_k[kxx,kyy,kzz] = real_part + 1j*imag_part

                    # fill the bottom plane of the delta_k array
                    # we do this ONLY if we fill up the upper plane
                    # we need to satisfy delta(-k) = delta*(k)
                    # k=(kx,ky,kz)---> -k=(-kx,-ky,-kz). For kz!=0 or kz!=middle
                    # the vector -k is not in memory, so we dont care
                    # thus, we only care when kz==0 or kz==middle
                    if kz==0 or kz==middle: #for these points: -kz=kz
                        if delta_k[kxx_m,kyy_m,kzz]==zero:
                            delta_k[kxx_m,kyy_m,kzz] = real_part - 1j*imag_part
                        if kxx_m==kxx and kyy_m==kyy: #when k=-k delta(k) should be real
                            delta_k[kxx,kyy,kzz] = amplitude + 1j*0.0

    # force this in case input Pk doesnt go to k=0
    delta_k[0,0,0] = zero

    return delta_k


def get_positions(seed, number_particles, bin_size):
    np.random.seed(seed)
    Rs = 2 * np.random.random(size = (number_particles,3)) - 1
    Rs = np.sign(Rs) * (1 - np.sqrt(np.abs(Rs)))
    
    return Rs * bin_size
#@numba.njit(fastmath=True)
def populate_field(rho, n_bins, box_size, density, seed):
    bin_size = box_size / n_bins
    cell_volume = bin_size**3
    print(cell_volume)
    mean_obj_per_cell = cell_volume * density
    rho *= mean_obj_per_cell / rho.mean()
    nonzero = (rho != 0).sum()
    sorted_rho = np.argsort(rho.ravel())[::-1]#[:nonzero+1]
    
    flat_rho = rho.flatten()[sorted_rho]
    grid_centers = np.array(np.unravel_index(sorted_rho, (n_bins, n_bins, n_bins))).T.astype(np.float32) * bin_size + 0.5 * bin_size
    np.random.seed(seed)
    
    number_objects = np.random.poisson(flat_rho, size=flat_rho.shape)
    

    
    displacements = get_positions(seed, number_objects.sum(), bin_size)

    coords = np.repeat(grid_centers, number_objects, axis=0) + displacements
    return (coords + box_size) % box_size
