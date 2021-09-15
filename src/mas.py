import jax
import jax.numpy as jnp
from jax.experimental import loops

@jax.jit
def cic_mas(delta,
            x, y, z, w, 
            n_part, 
            xmin, 
            ymin,
            zmin,
            box_size,
            n_bins,
            wrap):


    bin_size = box_size / n_bins
    inv_bin_size = 1. / bin_size

    def wrap_fun(input):
        ixp, iyp, izp, ddx, ddy, ddz, nbins = input
        subs_nbins = lambda x : x - nbins
        nothing = lambda x: x
        ixp = jax.lax.cond(ixp >= nbins, subs_nbins, nothing, ixp)
        iyp = jax.lax.cond(iyp >= nbins, subs_nbins, nothing, iyp)
        izp = jax.lax.cond(izp >= nbins, subs_nbins, nothing, izp)
        
        return (ixp, iyp, izp, ddx, ddy, ddz, nbins)

    def nowrap_fun(input):
        ixp, iyp, izp, ddx, ddy, ddz, nbins = input
        ret_zero = lambda _ : (0, 0.)
        nothing = lambda tup: tup
        ixp, ddx = jax.lax.cond(ixp >= nbins, ret_zero, nothing, (ixp, ddx))
        iyp, ddy = jax.lax.cond(iyp >= nbins, ret_zero, nothing, (iyp, ddy))
        izp, ddz = jax.lax.cond(izp >= nbins, ret_zero, nothing, (izp, ddz))

        return (ixp, iyp, izp, ddx, ddy, ddz, nbins)

    def assign_particle(input, xs):
        x, y, z, w = xs
        delta, xmin, ymin, zmin, n_bins, wrap =  input
        weight = w
        x_pos = (x - xmin) * inv_bin_size
        y_pos = (y - ymin) * inv_bin_size
        z_pos = (z - zmin) * inv_bin_size

        ix = jnp.int32(x_pos)
        iy = jnp.int32(y_pos)
        iz = jnp.int32(z_pos)

        ddx = x_pos - ix
        ddy = y_pos - iy
        ddz = z_pos - iz

        mdx = 1. - ddx
        mdy = 1. - ddy
        mdz = 1. - ddz

        ixp = ix + 1
        iyp = iy + 1
        izp = iz + 1

        

        #print(ixp, iyp, izp)       
        ixp, iyp, izp, ddx, ddy, ddz, n_bins = jax.lax.cond(wrap, wrap_fun, nowrap_fun, (ixp, iyp, izp, ddx, ddy, ddz, n_bins))
        #print(ixp, iyp, izp)

        delta = jax.ops.index_add(delta, (ix, iy, iz), mdx * mdy * mdz * weight)
        delta = jax.ops.index_add(delta, (ixp, iy, iz), ddx * mdy * mdz * weight)
        delta = jax.ops.index_add(delta, (ix, iyp, iz), mdx * ddy * mdz * weight)
        delta = jax.ops.index_add(delta, (ix, iy, izp), mdx * mdy * ddz * weight)

        delta = jax.ops.index_add(delta, (ixp, iyp, iz), ddx * ddy * mdz * weight)
        delta = jax.ops.index_add(delta, (ixp, iy, izp), ddx * mdy * ddz * weight)
        delta = jax.ops.index_add(delta, (ix, iyp, izp), mdx * ddy * ddz * weight)

        delta = jax.ops.index_add(delta, (ixp, iyp, izp), ddx * ddy * ddz * weight)

        return (delta, xmin, ymin, zmin, n_bins, wrap), (x, y, z, w)
    
    (delta, xmin, ymin, zmin, n_bins, wrap), (x, y, z, w) = jax.lax.scan(assign_particle, (delta, xmin, ymin, zmin, n_bins, wrap), (x, y, z, w))

    
    
    return delta
@jax.jit
def cic_mas_vec(delta,
            x, y, z, w, 
            n_part, 
            xmin, 
            ymin,
            zmin,
            box_size,
            n_bins,
            wrap):


    bin_size = box_size / n_bins
    inv_bin_size = 1. / bin_size

    xpos = (x - xmin) * inv_bin_size
    ypos = (y - ymin) * inv_bin_size
    zpos = (z - zmin) * inv_bin_size

    ix = jnp.int32(xpos)# % nbins
    iy = jnp.int32(ypos)# % nbins
    iz = jnp.int32(zpos)# % nbins

    ddx = xpos-ix
    ddy = ypos-iy
    ddz = zpos-iz

    ixp = ix + 1;
    iyp = iy + 1;
    izp = iz + 1;

    def true_wrap(input):
        ixp, iyp, izp, ddx, ddy, ddz, n_bins = input
        return (ixp + n_bins) % n_bins, (iyp + n_bins) % n_bins, (izp + n_bins) % n_bins,\
                ddx, ddy, ddz
    def false_wrap(input):
        ixp, iyp, izp, ddx, ddy, ddz, n_bins = input
        ixp = jnp.where(ixp >= n_bins, 0, ixp)
        ddx = jnp.where(ixp >= n_bins, 0, ddx)

        iyp = jnp.where(iyp >= n_bins, 0, iyp)
        ddy = jnp.where(iyp >= n_bins, 0, ddy)

        izp = jnp.where(izp >= n_bins, 0, izp)
        ddz = jnp.where(izp >= n_bins, 0, ddz)

        return ixp, iyp, izp, ddx, ddy, ddz

    ixp, iyp, izp, ddx, ddy, ddz = jax.lax.cond(wrap, true_wrap, false_wrap, (ixp, iyp, izp, ddx, ddy, ddz, n_bins))
    
    mdx = (1.0 - ddx)
    mdy = (1.0 - ddy)
    mdz = (1.0 - ddz)

    delta = jax.ops.index_add(delta, (ix, iy, iz), mdx * mdy * mdz * w)
    delta = jax.ops.index_add(delta, (ixp, iy, iz), ddx * mdy * mdz * w)
    delta = jax.ops.index_add(delta, (ix, iyp, iz), mdx * ddy * mdz * w)
    delta = jax.ops.index_add(delta, (ix, iy, izp), mdx * mdy * ddz * w)

    delta = jax.ops.index_add(delta, (ixp, iyp, iz), ddx * ddy * mdz * w)
    delta = jax.ops.index_add(delta, (ixp, iy, izp), ddx * mdy * ddz * w)
    delta = jax.ops.index_add(delta, (ix, iyp, izp), mdx * ddy * ddz * w)

    delta = jax.ops.index_add(delta, (ixp, iyp, izp), ddx * ddy * ddz * w)

    return delta

    

    
    
    return delta



