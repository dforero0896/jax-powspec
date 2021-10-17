import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from src.populate_field import populate_field, ezmock
from src.mas import cic_mas, cic_mas_vec
from src.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, s_edges_conv
import time
import proplot as pplt

n_bins = 256
n_part = 1000
box_size = 1000.
k_ny = jnp.pi * n_bins / box_size
particles = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), names=['x', 'y', 'z'], dtype='float32', delim_whitespace=True).values
mask = ((particles < box_size) & (particles > 0)).all(axis=1)
particles = jax.device_put(particles[mask].copy())
n_part = particles.shape[0]
print(n_part)
s_edges = jnp.linspace(0, 200, 51)



shot_noise = box_size**3 / n_part


w = jnp.ones(n_part)
s = time.time()
delta_r = jnp.zeros((n_bins, n_bins, n_bins))
delta_r = cic_mas_vec(delta_r,
                particles[:,0], particles[:,1], particles[:,2], w, 
                n_part, 
                particles[:,0].min(), particles[:,1].min(), particles[:,2].min(),
                box_size,
                n_bins,
                True)
print(f"MAS took {time.time() - s} s.", flush=True)

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

new_cat = ezmock((0.2, 1., 0.5), delta_r, n_bins, box_size, subkey)

delta = delta_r / delta_r.mean()
delta -= 1.



fig, ax = pplt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)

s, xi, _ = xi_vec(delta, box_size, s_edges_conv(n_bins, box_size, s_edges))
ax[2].plot(s, s**2*xi[:,0], label='input')


m = ax[0].imshow(delta.mean(axis=-1), vmin=-1, vmax=2)
ax[0].colorbar(m, loc='top')

w = jnp.ones(new_cat.shape[0])
delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
                new_cat[:,0], new_cat[:,1], new_cat[:,2], w, 
                n_part, 
                new_cat[:,0].min(), new_cat[:,1].min(), new_cat[:,2].min(),
                box_size,
                n_bins,
                True)
delta /= delta.mean()
delta -= 1.
s, xi, _ = xi_vec(delta, box_size, s_edges_conv(n_bins, box_size, s_edges))
ax[2].plot(s, s**2*xi[:,0], label='populated')
ax[2].legend()

m = ax[1].imshow(delta.mean(axis=-1), vmin=-1, vmax=2)
ax[1].colorbar(m, loc='top')


key, subkey = jax.random.split(key)
new_cat = populate_field(delta_r, n_bins, box_size, subkey)
w = jnp.ones(new_cat.shape[0])

delta = jnp.zeros((n_bins, n_bins, n_bins))
delta = cic_mas_vec(delta,
                new_cat[:,0], new_cat[:,1], new_cat[:,2], w, 
                n_part, 
                new_cat[:,0].min(), new_cat[:,1].min(), new_cat[:,2].min(),
                box_size,
                n_bins,
                True)
delta /= delta.mean()
delta -= 1.
s, xi, _ = xi_vec(delta, box_size, s_edges_conv(n_bins, box_size, s_edges))
ax[2].plot(s, s**2*xi[:,0], label='populated 2')






fig.savefig('plots/populate.png', dpi=300)